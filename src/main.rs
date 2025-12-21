// Copyright (c) 2025 Jan Holthuis <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

use binrw::{BinRead, BinWrite};
use clap::{Parser, Subcommand};
use rekordcrate::anlz::ANLZ;
use rekordcrate::pdb::ext::ExtPageType;
use rekordcrate::pdb::{
    DatabaseType, Header, PageContent, PageType, PlainPageType, Track, TrackId,
};

use rekordcrate::setting::Setting;
use rekordcrate::xml::Document;
use std::collections::HashSet;
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(author, version, about)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List the playlist tree from a Pioneer Database (`.PDB`) file.
    ListPlaylists {
        /// File to parse.
        #[arg(value_name = "PDB_FILE")]
        path: PathBuf,
    },
    /// Display settings from a Rekordbox device export.
    ListSettings {
        /// Path to parse.
        #[arg(value_name = "EXPORT_PATH")]
        path: PathBuf,
    },
    /// Export the playlists from a Pioneer Database (`.PDB`) file to M3U files.
    ExportPlaylists {
        /// File to parse.
        #[arg(value_name = "EXPORT_PATH")]
        path: PathBuf,
        /// Output directory to write M3U files to.
        #[arg(value_name = "OUTPUT_DIR")]
        output_dir: PathBuf,
    },
    /// Parse and dump a Rekordbox Analysis (`ANLZXXXX.DAT`) file.
    DumpANLZ {
        /// File to parse.
        #[arg(value_name = "ANLZ_FILE")]
        path: PathBuf,
    },
    /// Parse and dump a Pioneer Database (`.PDB`) file.
    DumpPDB {
        /// File to parse.
        #[arg(value_name = "PDB_FILE")]
        path: PathBuf,
        /// Database type: "plain" (export.pdb) or "ext" (exportExt.pdb). Tries to guess based on file name of not specified.
        #[arg(long, value_name = "DB_TYPE", value_parser = ["plain", "ext"])]
        db_type: Option<String>,
    },
    /// Parse and dump a Pioneer Settings (`*SETTING.DAT`) file.
    DumpSetting {
        /// File to parse.
        #[arg(value_name = "SETTING_FILE")]
        path: PathBuf,
    },
    /// Parse and dump a Pioneer XML (`*.xml`) file.
    DumpXML {
        /// File to parse.
        #[arg(value_name = "XML_FILE")]
        path: PathBuf,
    },
    /// Re-export a Pioneer Database (`.PDB`) file (read and write back).
    ReexportPDB {
        /// Input file to parse.
        #[arg(value_name = "INPUT_PDB_FILE")]
        inpath: PathBuf,
        /// Output file to write.
        #[arg(value_name = "OUTPUT_PDB_FILE")]
        outpath: PathBuf,
        /// Database type: "plain" (export.pdb) or "ext" (exportExt.pdb). Tries to guess based on file name if not specified.
        #[arg(long, value_name = "DB_TYPE", value_parser = ["plain", "ext"])]
        db_type: Option<String>,
        /// Comma-separated list of table types to include (e.g., "Tracks,Artists,Albums").
        /// If not specified, all tables are included.
        #[arg(long, value_name = "TABLES", value_delimiter = ',')]
        tables: Option<Vec<String>>,
    },
}

fn list_playlists(path: &PathBuf) -> rekordcrate::Result<()> {
    use rekordcrate::device::{Pdb, PlaylistNode};
    use std::collections::HashMap;

    let pdb = Pdb::open_from_path(path)?;
    let playlists = pdb.get_playlists()?;
    let tracks: HashMap<_, _> = pdb.get_tracks().map(|t| (t.id, t)).collect();

    fn print_node(pdb: &Pdb, tracks: &HashMap<TrackId, &Track>, node: &PlaylistNode, level: usize) {
        let indentation = "    ".repeat(level);
        match node {
            PlaylistNode::Folder(folder) => {
                println!("{}🗀 {}", indentation, folder.name);
                for child in &folder.children {
                    print_node(pdb, tracks, child, level + 1);
                }
            }
            PlaylistNode::Playlist(playlist) => {
                println!("{}🗎 {}", indentation, playlist.name);
                let mut entries: Vec<_> = pdb.get_playlist_entries(playlist.id).collect();
                entries.sort_by_key(|(index, _)| *index);
                for (index, track_id) in entries {
                    if let Some(track) = tracks.get(&track_id) {
                        println!("{}  ♫ {}: {}", indentation, index, track.offsets.title);
                    } else {
                        println!(
                            "{}  ♫ {}: <Track for {:?} not found>",
                            indentation, index, track_id
                        );
                    }
                }
            }
        }
    }

    for node in &playlists {
        print_node(&pdb, &tracks, node, 0);
    }

    Ok(())
}

fn export_playlists(path: &Path, output_dir: &PathBuf) -> rekordcrate::Result<()> {
    use rekordcrate::device::PlaylistNode;
    use rekordcrate::pdb::{Track, TrackId};
    use rekordcrate::DeviceExport;
    use std::collections::HashMap;
    use std::io::Write;

    let mut export = DeviceExport::new(path.into());
    export.load_pdb()?;
    let pdb = export.pdb().ok_or(rekordcrate::Error::NotLoadedError)?;
    let playlists = pdb.get_playlists()?;
    let tracks = pdb
        .get_tracks()
        .map(|track| (track.id, track))
        .collect::<HashMap<_, _>>();

    fn walk_tree(
        pdb: &rekordcrate::device::Pdb,
        tracks: &HashMap<TrackId, &Track>,
        node: PlaylistNode,
        path: &PathBuf,
        export_path: &Path,
    ) -> rekordcrate::Result<()> {
        match node {
            PlaylistNode::Folder(folder) => {
                folder.children.into_iter().try_for_each(|child| {
                    walk_tree(pdb, tracks, child, &path.join(&folder.name), export_path)
                })?;
            }
            PlaylistNode::Playlist(playlist) => {
                let mut playlist_entries: Vec<(u32, TrackId)> =
                    pdb.get_playlist_entries(playlist.id).collect();
                playlist_entries.sort_by_key(|entry| entry.0);

                std::fs::create_dir_all(path)?;
                let playlist_path = path.join(format!("{}.m3u", playlist.name));

                println!("{}", playlist_path.display());
                let mut file = std::fs::File::create(playlist_path)?;
                playlist_entries
                    .into_iter()
                    .filter_map(|(_index, id)| tracks.get(&id))
                    .try_for_each(|track| -> rekordcrate::Result<()> {
                        let track_path = track.offsets.file_path.clone().into_string()?;
                        Ok(writeln!(
                            &mut file,
                            "{}",
                            export_path
                                .canonicalize()?
                                .join(track_path.strip_prefix('/').unwrap_or(&track_path))
                                .display(),
                        )?)
                    })?;
            }
        };

        Ok(())
    }

    playlists
        .into_iter()
        .try_for_each(|node| walk_tree(pdb, &tracks, node, output_dir, export.get_path()))?;

    Ok(())
}

fn list_settings(path: &Path) -> rekordcrate::Result<()> {
    use rekordcrate::DeviceExport;

    let mut export = DeviceExport::new(path.into());
    export.load_settings();
    let settings = export.get_settings();

    print!("{}", settings);

    Ok(())
}

fn dump_anlz(path: &PathBuf) -> rekordcrate::Result<()> {
    let mut reader = std::fs::File::open(path)?;
    let anlz = ANLZ::read(&mut reader)?;
    println!("{:#?}", anlz);

    Ok(())
}

fn dump_pdb(path: &PathBuf, typ: DatabaseType) -> rekordcrate::Result<()> {
    let mut reader = std::fs::File::open(path)?;
    let header = Header::read_args(&mut reader, (typ,))?;

    println!("{:#?}", header);

    for (i, table) in header.tables.iter().enumerate() {
        println!("Table {}: {:?}", i, table.page_type);
        for page in header
            .read_pages(
                &mut reader,
                binrw::Endian::NATIVE,
                (&table.first_page, &table.last_page, typ),
            )
            .unwrap()
            .into_iter()
        {
            println!("  {:?}", page);
            match page.content {
                PageContent::Data(data_content) => {
                    for (_, row) in data_content.rows {
                        println!("      {:?}", row);
                    }
                }
                PageContent::Index(index_content) => {
                    println!("    {:?}", index_content);
                    for entry in index_content.entries {
                        println!("      {:?}", entry);
                    }
                }
                PageContent::Unknown => (),
            }
        }
    }

    Ok(())
}

fn dump_setting(path: &PathBuf) -> rekordcrate::Result<()> {
    let mut reader = std::fs::File::open(path)?;
    let setting = Setting::read(&mut reader)?;

    println!("{:#04x?}", setting);

    Ok(())
}

fn dump_xml(path: &PathBuf) -> rekordcrate::Result<()> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let document: Document = quick_xml::de::from_reader(reader).expect("failed to deserialize XML");
    println!("{:#?}", document);

    Ok(())
}

/// Table filter that can contain either Plain or Ext page types.
#[derive(Debug)]
enum TableFilter {
    Plain(HashSet<PlainPageType>),
    Ext(HashSet<ExtPageType>),
}

/// Parse the table filter from command line arguments.
/// Returns None if no filter is specified (include all tables).
/// Returns Some(TableFilter) with the table types to include based on db_type.
fn parse_table_filter(tables: &Option<Vec<String>>, db_type: DatabaseType) -> Option<TableFilter> {
    tables.as_ref().map(|table_names| match db_type {
        DatabaseType::Plain => {
            let filter: HashSet<PlainPageType> = table_names
                .iter()
                .filter_map(|name| match name.to_lowercase().as_str() {
                    "tracks" => Some(PlainPageType::Tracks),
                    "genres" => Some(PlainPageType::Genres),
                    "artists" => Some(PlainPageType::Artists),
                    "albums" => Some(PlainPageType::Albums),
                    "labels" => Some(PlainPageType::Labels),
                    "keys" => Some(PlainPageType::Keys),
                    "colors" => Some(PlainPageType::Colors),
                    "playlisttree" => Some(PlainPageType::PlaylistTree),
                    "playlistentries" => Some(PlainPageType::PlaylistEntries),
                    "historyplaylists" => Some(PlainPageType::HistoryPlaylists),
                    "historyentries" => Some(PlainPageType::HistoryEntries),
                    "artwork" => Some(PlainPageType::Artwork),
                    "columns" => Some(PlainPageType::Columns),
                    "menu" => Some(PlainPageType::Menu),
                    "history" => Some(PlainPageType::History),
                    unknown => {
                        eprintln!("Warning: Unknown plain table type '{}', ignoring.", unknown);
                        None
                    }
                })
                .collect();
            TableFilter::Plain(filter)
        }
        DatabaseType::Ext => {
            let filter: HashSet<ExtPageType> = table_names
                .iter()
                .filter_map(|name| match name.to_lowercase().as_str() {
                    "tag" => Some(ExtPageType::Tag),
                    "tracktag" => Some(ExtPageType::TrackTag),
                    unknown => {
                        eprintln!("Warning: Unknown ext table type '{}', ignoring.", unknown);
                        None
                    }
                })
                .collect();
            TableFilter::Ext(filter)
        }
    })
}

fn reexport_pdb(
    inpath: &PathBuf,
    outpath: &PathBuf,
    db_type: DatabaseType,
    tables: &Option<Vec<String>>,
) -> rekordcrate::Result<()> {
    let table_filter = parse_table_filter(tables, db_type);

    let mut reader = std::fs::File::open(inpath)?;
    let header = Header::read_args(&mut reader, (db_type,))?;
    let mut writer = std::fs::OpenOptions::new().write(true).open(outpath)?;

    for table in &header.tables {
        let process_table = match (&table_filter, &table.page_type) {
            (None, _) => true, // No filter, process all tables
            (Some(TableFilter::Plain(filter)), PageType::Plain(plain_type)) => {
                filter.contains(plain_type)
            }
            (Some(TableFilter::Ext(filter)), PageType::Ext(ext_type)) => filter.contains(ext_type),

            // these two shouldn't happen
            (Some(TableFilter::Plain(_)), PageType::Ext(_)) => false,
            (Some(TableFilter::Ext(_)), PageType::Plain(_)) => false,

            // don't process unknown tables
            (Some(_), PageType::Unknown(_)) => false,
        };

        if !process_table {
            continue;
        }

        // Read all pages for this table
        let pages = header.read_pages(
            &mut reader,
            binrw::Endian::NATIVE,
            (&table.first_page, &table.last_page, db_type),
        )?;

        // Write each page back at the correct offset
        for page in pages {
            let page_offset = page.header.page_index.offset(header.page_size);
            writer.seek(SeekFrom::Start(page_offset))?;
            page.write_args(&mut writer, (header.page_size,))?;
        }
    }

    println!(
        "Successfully re-exported PDB from {} to {}",
        inpath.display(),
        outpath.display()
    );
    Ok(())
}

fn guess_db_type(path: &Path, db_type: Option<&str>) -> Option<DatabaseType> {
    let db_type_cli = db_type.map(|str| match str {
        "plain" => DatabaseType::Plain,
        "ext" => DatabaseType::Ext,
        invalid => unreachable!("invalid flag {invalid}, should have already been checked by clap"),
    });
    let file_name = match path.file_name() {
        None => {
            eprintln!("{} not a file!", path.display());
            return None; // TODO(Swiftb0y): turn this into a proper error
        }
        Some(file_name) => file_name,
    };
    let db_type_file = if file_name == "export.pdb" {
        Some(DatabaseType::Plain)
    } else if file_name == "exportExt.pdb" {
        Some(DatabaseType::Ext)
    } else {
        None
    };
    let db_type = match (db_type_cli, db_type_file) {
        (None, None) => {
            eprintln!("no DB_TYPE supplied nor could it be guessed!");
            return None; // TODO(Swiftb0y): turn this into a proper error
        }
        (None, Some(guess)) | (Some(guess), None) => guess,
        (Some(db_type_cli), Some(db_type_file)) if db_type_cli == db_type_file => db_type_cli,
        (Some(db_type_cli), Some(db_type_file)) => {
            eprintln!("Warning: passed {db_type_cli:?}, but found {db_type_file:?} from file name, using {db_type_cli:?}!");
            db_type_cli
        }
    };
    Some(db_type)
}

fn main() -> rekordcrate::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::ListPlaylists { path } => list_playlists(path),
        Commands::ListSettings { path } => list_settings(path),
        Commands::ExportPlaylists { path, output_dir } => export_playlists(path, output_dir),
        Commands::DumpPDB { path, db_type } => {
            let db_type = match guess_db_type(path, db_type.as_deref()) {
                Some(db_type) => db_type,
                None => return Ok(()), // TODO(Swiftb0y): turn into proper error;
            };
            dump_pdb(path, db_type)
        }
        Commands::DumpANLZ { path } => dump_anlz(path),
        Commands::DumpSetting { path } => dump_setting(path),
        Commands::DumpXML { path } => dump_xml(path),
        Commands::ReexportPDB {
            inpath,
            outpath,
            db_type,
            tables,
        } => {
            let db_type = match guess_db_type(inpath, db_type.as_deref()) {
                Some(db_type) => db_type,
                None => return Ok(()), // TODO(Swiftb0y): turn into proper error;
            };
            reexport_pdb(inpath, outpath, db_type, tables)
        }
    }
}
