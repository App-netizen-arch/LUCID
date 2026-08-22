//! Graph Module for Gestalt-MVP
//! 
//! Manages gesture prototypes and meaning mappings.
//! Uses SQLite (rusqlite) for persistence.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::message_bus::GestureId;

/// A gesture prototype with multiple enrolled exemplars
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GesturePrototype {
    pub id: GestureId,
    pub label: String,
    pub trajectories: Vec<Trajectory>,
    pub exemplar_count: usize,
}

/// A single trajectory (normalized landmarks over time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub frames: Vec<Frame>,
    pub duration_s: f64,
    pub frame_count: usize,
}

/// A single frame with 21 landmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub t: f64,
    pub landmarks: Vec<[f32; 3]>, // 21 points × [x, y, z]
}

/// Mapping from gesture to meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mapping {
    pub gesture_id: GestureId,
    pub meaning: String,
    pub phrase_template: String,
    pub active: bool,
}

impl GesturePrototype {
    pub fn new(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            trajectories: Vec::new(),
            exemplar_count: 0,
        }
    }

    pub fn add_exemplar(&mut self, trajectory: Trajectory) {
        self.trajectories.push(trajectory);
        self.exemplar_count = self.trajectories.len();
    }

    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

impl Mapping {
    pub fn new(gesture_id: &str, meaning: &str, phrase_template: &str) -> Self {
        Self {
            gesture_id: gesture_id.to_string(),
            meaning: meaning.to_string(),
            phrase_template: phrase_template.to_string(),
            active: true,
        }
    }

    /// Get the frozen phrase templates from Section 8
    pub fn get_frozen_templates() -> Vec<Mapping> {
        vec![
            Mapping::new("G1", "Water request", "I would like some water, please."),
            Mapping::new("G2", "Help request", "Could you help me, please?"),
            Mapping::new("G3", "Affirmative", "Yes."),
            Mapping::new("G4", "Negative", "No."),
            Mapping::new("G5", "Food request", "I would like some food, please."),
            Mapping::new("G6", "Pain report", "I am in pain. Please help me."),
            Mapping::new("G7", "Stop command", "Stop. Please stop."),
            Mapping::new("G8", "More request", "I would like more, please."),
            Mapping::new("G9", "Wait request", "Please wait a moment."),
            Mapping::new("G10", "Bathroom request", "I need to use the bathroom."),
        ]
    }
}

/// Graph database wrapper
pub struct GestureGraph {
    conn: Connection,
}

impl GestureGraph {
    /// Create or open the graph database
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        
        // Initialize tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS gestures (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                exemplar_count INTEGER DEFAULT 0
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture_id TEXT NOT NULL,
                json_data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (gesture_id) REFERENCES gestures(id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS mappings (
                gesture_id TEXT PRIMARY KEY,
                meaning TEXT NOT NULL,
                phrase_template TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                FOREIGN KEY (gesture_id) REFERENCES gestures(id)
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    /// Create an in-memory database (for testing)
    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        
        // Same table creation as above
        conn.execute(
            "CREATE TABLE gestures (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                exemplar_count INTEGER DEFAULT 0
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture_id TEXT NOT NULL,
                json_data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (gesture_id) REFERENCES gestures(id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE mappings (
                gesture_id TEXT PRIMARY KEY,
                meaning TEXT NOT NULL,
                phrase_template TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                FOREIGN KEY (gesture_id) REFERENCES gestures(id)
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    /// Initialize with frozen phrase templates
    pub fn initialize_mappings(&self) -> Result<(), rusqlite::Error> {
        let templates = Mapping::get_frozen_templates();
        
        for mapping in templates {
            self.conn.execute(
                "INSERT OR REPLACE INTO mappings (gesture_id, meaning, phrase_template, active)
                 VALUES (?1, ?2, ?3, ?4)",
                params![mapping.gesture_id, mapping.meaning, mapping.phrase_template, 
                       if mapping.active { 1 } else { 0 }],
            )?;
        }

        Ok(())
    }

    /// Add a gesture prototype
    pub fn add_gesture(&self, prototype: &GesturePrototype) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT OR REPLACE INTO gestures (id, label, exemplar_count)
             VALUES (?1, ?2, ?3)",
            params![prototype.id, prototype.label, prototype.exemplar_count],
        )?;

        Ok(())
    }

    /// Add a trajectory exemplar to a gesture
    pub fn add_trajectory(&self, gesture_id: &str, trajectory: &Trajectory) 
                          -> Result<i64, rusqlite::Error> {
        let json_data = serde_json::to_string(trajectory)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(e.into()))?;

        let id = self.conn.execute(
            "INSERT INTO trajectories (gesture_id, json_data) VALUES (?1, ?2)",
            params![gesture_id, json_data],
        )?;

        // Update exemplar count
        self.conn.execute(
            "UPDATE gestures SET exemplar_count = exemplar_count + 1 WHERE id = ?1",
            params![gesture_id],
        )?;

        Ok(id as i64)
    }

    /// Get all trajectories for a gesture
    pub fn get_trajectories(&self, gesture_id: &str) -> Result<Vec<Trajectory>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT json_data FROM trajectories WHERE gesture_id = ?1"
        )?;

        let trajectories = stmt.query_map(params![gesture_id], |row| {
            let json_str: String = row.get(0)?;
            serde_json::from_str::<Trajectory>(&json_str)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0))
        })?
        .filter_map(|r| r.ok())
        .collect();

        Ok(trajectories)
    }

    /// Get all gesture prototypes
    pub fn get_all_gestures(&self) -> Result<Vec<GesturePrototype>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, label, exemplar_count FROM gestures"
        )?;

        let gestures = stmt.query_map([], |row| {
            Ok(GesturePrototype {
                id: row.get(0)?,
                label: row.get(1)?,
                trajectories: Vec::new(), // Load separately
                exemplar_count: row.get(2)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

        Ok(gestures)
    }

    /// Get mapping for a gesture
    pub fn get_mapping(&self, gesture_id: &str) -> Result<Option<Mapping>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT gesture_id, meaning, phrase_template, active 
             FROM mappings WHERE gesture_id = ?1"
        )?;

        let mapping = stmt.query_row(params![gesture_id], |row| {
            Ok(Mapping {
                gesture_id: row.get(0)?,
                meaning: row.get(1)?,
                phrase_template: row.get(2)?,
                active: row.get::<_, i32>(3)? == 1,
            })
        });

        match mapping {
            Ok(m) => Ok(Some(m)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Update mapping (for correction handling)
    pub fn update_mapping(&self, gesture_id: &str, phrase_template: &str) 
                          -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "UPDATE mappings SET phrase_template = ?1 WHERE gesture_id = ?2",
            params![phrase_template, gesture_id],
        )?;

        Ok(())
    }

    /// Get exemplar count for a gesture
    pub fn get_exemplar_count(&self, gesture_id: &str) -> Result<usize, rusqlite::Error> {
        let count: i32 = self.conn.query_row(
            "SELECT exemplar_count FROM gestures WHERE id = ?1",
            params![gesture_id],
            |row| row.get(0),
        )?;

        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gesture_prototype_creation() {
        let mut proto = GesturePrototype::new("G1", "Water");
        assert_eq!(proto.id, "G1");
        assert_eq!(proto.exemplar_count, 0);

        let trajectory = Trajectory {
            frames: vec![],
            duration_s: 1.0,
            frame_count: 0,
        };
        proto.add_exemplar(trajectory);
        assert_eq!(proto.exemplar_count, 1);
    }

    #[test]
    fn test_frozen_templates() {
        let templates = Mapping::get_frozen_templates();
        assert_eq!(templates.len(), 10);
        
        let g1 = templates.iter().find(|m| m.gesture_id == "G1").unwrap();
        assert_eq!(g1.phrase_template, "I would like some water, please.");
    }

    #[test]
    fn test_graph_database() {
        let graph = GestureGraph::open_in_memory().unwrap();
        graph.initialize_mappings().unwrap();

        let mut proto = GesturePrototype::new("G1", "Water");
        proto.add_exemplar(Trajectory {
            frames: vec![],
            duration_s: 1.0,
            frame_count: 0,
        });
        
        graph.add_gesture(&proto).unwrap();
        graph.add_trajectory("G1", &proto.trajectories[0]).unwrap();

        let mapping = graph.get_mapping("G1").unwrap().unwrap();
        assert_eq!(mapping.phrase_template, "I would like some water, please.");

        let count = graph.get_exemplar_count("G1").unwrap();
        assert_eq!(count, 1);
    }
}
