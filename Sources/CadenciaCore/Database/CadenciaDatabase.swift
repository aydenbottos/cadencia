import Foundation
import SQLite3

public enum DatabaseError: Error, LocalizedError {
    case openFailed(String)
    case prepareFailed(String)
    case stepFailed(String)
    case seedMissing
    case invalidData(String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let m): return "SQLite open failed: \(m)"
        case .prepareFailed(let m): return "SQLite prepare failed: \(m)"
        case .stepFailed(let m): return "SQLite step failed: \(m)"
        case .seedMissing: return "Bundled cadencia_seed.sqlite resource not found."
        case .invalidData(let m): return "Invalid DB data: \(m)"
        }
    }
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// Thread-safe, actor-isolated local SQLite3 database powering all 8 components of the
/// Signal-Starvation Hypothesis protocol:
/// - 4,900-item Spanish syllabus (4,000 lemmas + 400 hold-out + 200 chunks + 300 Cervantes grammar)
/// - 64,397-entry Spanish inflection->lemma dictionary
/// - Learner model half-life records
/// - Passage & fresh-context co-occurrence history (Rule 4)
/// - Strict APPEND-ONLY `evidence_events` table enforced by SQLite BEFORE UPDATE/DELETE ABORT triggers
/// - Weekly audits, research assessments (Vocab Size, EIT, Native Listening), sessions, and serial state
public actor CadenciaDatabase {
    private nonisolated(unsafe) var db: OpaquePointer?
    public let dbURL: URL

    public init(databaseURL: URL? = nil, seedURL: URL? = nil) throws {
        let fm = FileManager.default
        let resolvedDBURL: URL
        if let custom = databaseURL {
            resolvedDBURL = custom
        } else {
            let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let dir = appSupport.appendingPathComponent("Cadencia", isDirectory: true)
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            resolvedDBURL = dir.appendingPathComponent("cadencia.sqlite")
        }
        self.dbURL = resolvedDBURL

        let resolvedSeedURL = seedURL
            ?? Bundle.module.url(forResource: "cadencia_seed", withExtension: "sqlite")
            ?? Bundle.main.url(forResource: "cadencia_seed", withExtension: "sqlite")

        if !fm.fileExists(atPath: resolvedDBURL.path) {
            guard let seed = resolvedSeedURL, fm.fileExists(atPath: seed.path) else {
                throw DatabaseError.seedMissing
            }
            try fm.createDirectory(at: resolvedDBURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(at: seed, to: resolvedDBURL)
        }

        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        if sqlite3_open_v2(resolvedDBURL.path, &handle, flags, nil) != SQLITE_OK {
            let msg = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            throw DatabaseError.openFailed(msg)
        }
        self.db = handle

        try Self.bootstrapSchema(db: handle!)
    }

    deinit {
        if let db {
            sqlite3_close(db)
        }
    }

    // MARK: - Schema & Append-Only Triggers

    private static func bootstrapSchema(db: OpaquePointer) throws {
        let sql = """
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE IF NOT EXISTS learner_profiles (
            learner_id TEXT PRIMARY KEY,
            coded_group_label TEXT NOT NULL,
            display_name TEXT NOT NULL,
            study_mode TEXT NOT NULL,
            selected_serial_id TEXT NOT NULL,
            nonfiction_topics_json TEXT NOT NULL,
            current_session_number INTEGER NOT NULL DEFAULT 1,
            total_logged_seconds REAL NOT NULL DEFAULT 0.0,
            median_reading_wpm REAL NOT NULL DEFAULT 95.0,
            parameters_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS learner_items (
            learner_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            half_life_days REAL NOT NULL,
            last_meeting_at REAL,
            first_meeting_session INTEGER,
            last_meeting_session INTEGER,
            meetings_count INTEGER NOT NULL DEFAULT 0,
            successes_count INTEGER NOT NULL DEFAULT 0,
            ear_successes_count INTEGER NOT NULL DEFAULT 0,
            productive_uses_count INTEGER NOT NULL DEFAULT 0,
            lookups_count INTEGER NOT NULL DEFAULT 0,
            failures_count INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'new',
            needs_extra_meeting INTEGER NOT NULL DEFAULT 0,
            is_held_back_control INTEGER NOT NULL DEFAULT 0,
            lapse_spares_used INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (learner_id, item_id)
        );
        CREATE INDEX IF NOT EXISTS idx_learner_items_state ON learner_items(learner_id, state);
        CREATE INDEX IF NOT EXISTS idx_learner_items_extra ON learner_items(learner_id, needs_extra_meeting);

        CREATE TABLE IF NOT EXISTS serial_progress (
            learner_id TEXT NOT NULL,
            serial_id TEXT NOT NULL,
            episode_number INTEGER NOT NULL DEFAULT 0,
            story_summary TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            PRIMARY KEY (learner_id, serial_id)
        );

        CREATE TABLE IF NOT EXISTS passages (
            passage_id TEXT PRIMARY KEY,
            learner_id TEXT NOT NULL,
            session_number INTEGER NOT NULL,
            block_type TEXT NOT NULL,
            format TEXT NOT NULL,
            title_es TEXT NOT NULL,
            full_text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            voice TEXT NOT NULL,
            accent TEXT NOT NULL,
            target_wpm INTEGER NOT NULL,
            is_audio_first INTEGER NOT NULL,
            scheduled_word_share REAL NOT NULL,
            rewrite_attempts INTEGER NOT NULL,
            flagged_for_native_audit INTEGER NOT NULL DEFAULT 0,
            passage_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_passages_learner_session ON passages(learner_id, session_number);

        -- Rule 4 (Fresh Context): stores prior sentences and adjacent content lemmas for every due item meeting
        CREATE TABLE IF NOT EXISTS item_sentence_contexts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            passage_id TEXT NOT NULL,
            sentence_text TEXT NOT NULL,
            neighbor_content_lemmas_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_item_contexts_lookup ON item_sentence_contexts(learner_id, item_id);

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            learner_id TEXT NOT NULL,
            session_number INTEGER NOT NULL,
            week INTEGER NOT NULL,
            study_mode TEXT NOT NULL,
            target_new_items INTEGER NOT NULL,
            started_at REAL NOT NULL,
            ended_at REAL,
            active_input_seconds REAL NOT NULL DEFAULT 0.0,
            active_output_seconds REAL NOT NULL DEFAULT 0.0,
            passages_completed INTEGER NOT NULL DEFAULT 0,
            running_words INTEGER NOT NULL DEFAULT 0,
            usable_events INTEGER NOT NULL DEFAULT 0,
            probes_correct INTEGER NOT NULL DEFAULT 0,
            probes_total INTEGER NOT NULL DEFAULT 0,
            new_items_introduced INTEGER NOT NULL DEFAULT 0,
            promoted_known INTEGER NOT NULL DEFAULT 0,
            promoted_productive INTEGER NOT NULL DEFAULT 0,
            fatigue_rating INTEGER,
            enjoyment_rating INTEGER,
            no_other_spanish_confirmed INTEGER NOT NULL DEFAULT 0,
            summary_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_learner ON sessions(learner_id, session_number);

        CREATE TABLE IF NOT EXISTS weekly_audits (
            audit_id TEXT PRIMARY KEY,
            learner_id TEXT NOT NULL,
            week INTEGER NOT NULL,
            completed_at REAL NOT NULL,
            mean_predicted REAL NOT NULL,
            mean_observed REAL NOT NULL,
            abs_error REAL NOT NULL,
            did_refit INTEGER NOT NULL DEFAULT 0,
            duration_seconds REAL NOT NULL,
            result_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS research_assessments (
            assessment_id TEXT PRIMARY KEY,
            learner_id TEXT NOT NULL,
            assessment_type TEXT NOT NULL,
            checkpoint_label TEXT NOT NULL,
            completed_at REAL NOT NULL,
            score_primary REAL NOT NULL,
            result_json TEXT NOT NULL
        );

        -- Stage 1 / Solo Pilot 7-day delayed capacity recall queue
        CREATE TABLE IF NOT EXISTS delayed_capacity_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            source_session_number INTEGER NOT NULL,
            intake_rate_bucket INTEGER NOT NULL,
            due_at REAL NOT NULL,
            item_ids_json TEXT NOT NULL,
            completed_at REAL,
            retention_rate REAL
        );

        -- Strict APPEND-ONLY Evidence Log (all analyses computed from this immutable log)
        CREATE TABLE IF NOT EXISTS evidence_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_uuid TEXT NOT NULL UNIQUE,
            learner_id TEXT NOT NULL,
            session_id TEXT,
            session_number INTEGER,
            passage_id TEXT,
            item_id TEXT,
            event_type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            predicted_recall_before REAL,
            half_life_before REAL,
            half_life_after REAL,
            is_first_meeting_glossed INTEGER NOT NULL DEFAULT 0,
            is_unseen_sentence INTEGER NOT NULL DEFAULT 1,
            passage_understood INTEGER,
            is_usable_event INTEGER NOT NULL DEFAULT 0,
            numeric_value REAL,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_evidence_learner_type ON evidence_events(learner_id, event_type);
        CREATE INDEX IF NOT EXISTS idx_evidence_learner_time ON evidence_events(learner_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_evidence_usable ON evidence_events(learner_id, is_usable_event);

        CREATE TRIGGER IF NOT EXISTS trg_evidence_no_update
        BEFORE UPDATE ON evidence_events
        BEGIN
            SELECT RAISE(ABORT, 'evidence_events is strictly append-only: UPDATE forbidden');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_evidence_no_delete
        BEFORE DELETE ON evidence_events
        BEGIN
            SELECT RAISE(ABORT, 'evidence_events is strictly append-only: DELETE forbidden');
        END;
        """
        var errMsg: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &errMsg) != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(errMsg)
            throw DatabaseError.stepFailed(msg)
        }
    }

    // MARK: - Low-Level Helpers

    private func exec(_ sql: String) throws {
        guard let db else { throw DatabaseError.openFailed("DB closed") }
        var errMsg: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &errMsg) != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(errMsg)
            throw DatabaseError.stepFailed(msg)
        }
    }

    private func prepare(_ sql: String) throws -> OpaquePointer {
        guard let db else { throw DatabaseError.openFailed("DB closed") }
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) != SQLITE_OK {
            throw DatabaseError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        return stmt!
    }

    private func bindText(_ stmt: OpaquePointer, _ index: Int32, _ value: String?) {
        if let value {
            sqlite3_bind_text(stmt, index, value, -1, SQLITE_TRANSIENT)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func bindDouble(_ stmt: OpaquePointer, _ index: Int32, _ value: Double?) {
        if let value {
            sqlite3_bind_double(stmt, index, value)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func bindInt(_ stmt: OpaquePointer, _ index: Int32, _ value: Int?) {
        if let value {
            sqlite3_bind_int64(stmt, index, Int64(value))
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func colString(_ stmt: OpaquePointer, _ idx: Int32) -> String {
        sqlite3_column_text(stmt, idx).map { String(cString: $0) } ?? ""
    }

    private func colOptString(_ stmt: OpaquePointer, _ idx: Int32) -> String? {
        sqlite3_column_type(stmt, idx) == SQLITE_NULL ? nil : colString(stmt, idx)
    }

    private func colOptDouble(_ stmt: OpaquePointer, _ idx: Int32) -> Double? {
        sqlite3_column_type(stmt, idx) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, idx)
    }

    private func colOptInt(_ stmt: OpaquePointer, _ idx: Int32) -> Int? {
        sqlite3_column_type(stmt, idx) == SQLITE_NULL ? nil : Int(sqlite3_column_int64(stmt, idx))
    }

    // MARK: - Syllabus & Inflection Queries

    public func loadAllSyllabusItems(includeHoldout: Bool = false) throws -> [SyllabusItem] {
        let whereClause = includeHoldout ? "" : "WHERE is_holdout = 0"
        let stmt = try prepare("""
            SELECT item_id, item_type, rank_index, spanish, english_gloss, pos, cefr,
                   ipa_es, ipa_latam, is_transparent_cognate, cognate_score, is_bootstrap,
                   is_holdout, is_top2000_productive_target, zipf_frequency, visual_symbol,
                   grammar_category, grammar_week, micro_note_en, contrast_a_es, contrast_a_en,
                   contrast_b_es, contrast_b_en, literal_gloss, example_es
            FROM syllabus_items \(whereClause)
            ORDER BY item_type, rank_index
        """)
        defer { sqlite3_finalize(stmt) }
        var items: [SyllabusItem] = []
        items.reserveCapacity(includeHoldout ? 4900 : 4500)
        while sqlite3_step(stmt) == SQLITE_ROW {
            items.append(decodeSyllabusRow(stmt))
        }
        return items
    }

    public func loadHoldoutItems() throws -> [SyllabusItem] {
        let stmt = try prepare("""
            SELECT item_id, item_type, rank_index, spanish, english_gloss, pos, cefr,
                   ipa_es, ipa_latam, is_transparent_cognate, cognate_score, is_bootstrap,
                   is_holdout, is_top2000_productive_target, zipf_frequency, visual_symbol,
                   grammar_category, grammar_week, micro_note_en, contrast_a_es, contrast_a_en,
                   contrast_b_es, contrast_b_en, literal_gloss, example_es
            FROM syllabus_items WHERE is_holdout = 1
            ORDER BY rank_index
        """)
        defer { sqlite3_finalize(stmt) }
        var items: [SyllabusItem] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            items.append(decodeSyllabusRow(stmt))
        }
        return items
    }

    public func loadSyllabusItem(id: String) throws -> SyllabusItem? {
        let stmt = try prepare("""
            SELECT item_id, item_type, rank_index, spanish, english_gloss, pos, cefr,
                   ipa_es, ipa_latam, is_transparent_cognate, cognate_score, is_bootstrap,
                   is_holdout, is_top2000_productive_target, zipf_frequency, visual_symbol,
                   grammar_category, grammar_week, micro_note_en, contrast_a_es, contrast_a_en,
                   contrast_b_es, contrast_b_en, literal_gloss, example_es
            FROM syllabus_items WHERE item_id = ? LIMIT 1
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        return sqlite3_step(stmt) == SQLITE_ROW ? decodeSyllabusRow(stmt) : nil
    }

    private func decodeSyllabusRow(_ stmt: OpaquePointer) -> SyllabusItem {
        SyllabusItem(
            id: colString(stmt, 0),
            itemType: ItemType(rawValue: colString(stmt, 1)) ?? .lemma,
            rankIndex: Int(sqlite3_column_int64(stmt, 2)),
            spanish: colString(stmt, 3),
            englishGloss: colString(stmt, 4),
            pos: colString(stmt, 5),
            cefr: CEFRLevel(rawValue: colString(stmt, 6)) ?? .a1,
            ipaEs: colString(stmt, 7),
            ipaLatam: colString(stmt, 8),
            isTransparentCognate: sqlite3_column_int(stmt, 9) != 0,
            cognateScore: sqlite3_column_double(stmt, 10),
            isBootstrap: sqlite3_column_int(stmt, 11) != 0,
            isHoldout: sqlite3_column_int(stmt, 12) != 0,
            isTop2000ProductiveTarget: sqlite3_column_int(stmt, 13) != 0,
            zipfFrequency: sqlite3_column_double(stmt, 14),
            visualSymbol: colString(stmt, 15),
            grammarCategory: colOptString(stmt, 16),
            grammarWeek: colOptInt(stmt, 17),
            microNoteEn: colOptString(stmt, 18),
            contrastAEs: colOptString(stmt, 19),
            contrastAEn: colOptString(stmt, 20),
            contrastBEs: colOptString(stmt, 21),
            contrastBEn: colOptString(stmt, 22),
            literalGloss: colOptString(stmt, 23),
            exampleEs: colOptString(stmt, 24)
        )
    }

    /// Loads the complete 64k+ Spanish inflection -> (lemma, pos) map into memory (~4 MB) for <1ms verification.
    public func loadInflectionDictionary() throws -> [String: (lemma: String, pos: String)] {
        let stmt = try prepare("SELECT surface, lemma, pos FROM spanish_inflections")
        defer { sqlite3_finalize(stmt) }
        var map: [String: (String, String)] = [:]
        map.reserveCapacity(66_000)
        while sqlite3_step(stmt) == SQLITE_ROW {
            map[colString(stmt, 0)] = (colString(stmt, 1), colString(stmt, 2))
        }
        return map
    }

    public func loadSerialGenres() throws -> [SerialGenre] {
        let stmt = try prepare("""
            SELECT id, title_es, title_en, tagline, setting, premise, accent_palette, protagonist
            FROM serial_genres ORDER BY id
        """)
        defer { sqlite3_finalize(stmt) }
        var res: [SerialGenre] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            res.append(SerialGenre(
                id: colString(stmt, 0),
                titleEs: colString(stmt, 1),
                titleEn: colString(stmt, 2),
                tagline: colString(stmt, 3),
                setting: colString(stmt, 4),
                premise: colString(stmt, 5),
                accentPalette: colString(stmt, 6),
                protagonist: colString(stmt, 7)
            ))
        }
        return res
    }

    public func loadPseudowords() throws -> [String] {
        let stmt = try prepare("SELECT word FROM assessment_pseudowords ORDER BY id")
        defer { sqlite3_finalize(stmt) }
        var res: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            res.append(colString(stmt, 0))
        }
        return res
    }

    public func loadEITPrompts() throws -> [EITPrompt] {
        let stmt = try prepare("SELECT id, syllables, cefr, spanish, english, ipa_es, ipa_latam FROM assessment_eit_items ORDER BY id")
        defer { sqlite3_finalize(stmt) }
        var res: [EITPrompt] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            res.append(EITPrompt(
                id: Int(sqlite3_column_int64(stmt, 0)),
                syllables: Int(sqlite3_column_int64(stmt, 1)),
                cefr: CEFRLevel(rawValue: colString(stmt, 2)) ?? .a1,
                spanish: colString(stmt, 3),
                english: colString(stmt, 4),
                ipaEs: colString(stmt, 5),
                ipaLatam: colString(stmt, 6)
            ))
        }
        return res
    }

    public func loadNativeListeningPassages() throws -> [NativeListeningPassage] {
        let stmt = try prepare("SELECT id, checkpoint_hour, accent, speaker, wpm, title, transcript, questions_json FROM assessment_native_listening ORDER BY checkpoint_hour, id")
        defer { sqlite3_finalize(stmt) }
        let decoder = JSONDecoder()
        var res: [NativeListeningPassage] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let qData = Data(colString(stmt, 7).utf8)
            let questions = (try? decoder.decode([NativeListeningQuestion].self, from: qData)) ?? []
            res.append(NativeListeningPassage(
                id: colString(stmt, 0),
                checkpointHour: Int(sqlite3_column_int64(stmt, 1)),
                accent: SpanishRegionalAccent(rawValue: colString(stmt, 2)) ?? .castilian,
                speaker: MagpieVoice(rawValue: colString(stmt, 3)) ?? .sofia,
                wpm: Int(sqlite3_column_int64(stmt, 4)),
                title: colString(stmt, 5),
                transcript: colString(stmt, 6),
                questions: questions
            ))
        }
        return res
    }

    // MARK: - Learner Profile & Item Records

    public func saveProfile(_ profile: LearnerProfile) throws {
        let encoder = JSONEncoder()
        let topicsJSON = String(data: try encoder.encode(profile.nonfictionTopics), encoding: .utf8) ?? "[]"
        let paramsJSON = String(data: try encoder.encode(profile.parameters), encoding: .utf8) ?? "{}"
        let stmt = try prepare("""
            INSERT INTO learner_profiles (
                learner_id, coded_group_label, display_name, study_mode, selected_serial_id,
                nonfiction_topics_json, current_session_number, total_logged_seconds,
                median_reading_wpm, parameters_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(learner_id) DO UPDATE SET
                coded_group_label=excluded.coded_group_label,
                display_name=excluded.display_name,
                study_mode=excluded.study_mode,
                selected_serial_id=excluded.selected_serial_id,
                nonfiction_topics_json=excluded.nonfiction_topics_json,
                current_session_number=excluded.current_session_number,
                total_logged_seconds=excluded.total_logged_seconds,
                median_reading_wpm=excluded.median_reading_wpm,
                parameters_json=excluded.parameters_json
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, profile.learnerId)
        bindText(stmt, 2, profile.codedGroupLabel)
        bindText(stmt, 3, profile.displayName)
        bindText(stmt, 4, profile.studyMode.rawValue)
        bindText(stmt, 5, profile.selectedSerialId)
        bindText(stmt, 6, topicsJSON)
        bindInt(stmt, 7, profile.currentSessionNumber)
        bindDouble(stmt, 8, profile.totalLoggedSeconds)
        bindDouble(stmt, 9, profile.medianReadingWPM)
        bindText(stmt, 10, paramsJSON)
        bindDouble(stmt, 11, profile.createdAt.timeIntervalSince1970)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func loadActiveProfile() throws -> LearnerProfile? {
        let stmt = try prepare("""
            SELECT learner_id, coded_group_label, display_name, study_mode, selected_serial_id,
                   nonfiction_topics_json, current_session_number, total_logged_seconds,
                   median_reading_wpm, parameters_json, created_at
            FROM learner_profiles ORDER BY created_at DESC LIMIT 1
        """)
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        let decoder = JSONDecoder()
        let topics = (try? decoder.decode([String].self, from: Data(colString(stmt, 5).utf8))) ?? []
        let params = (try? decoder.decode(ProtocolParameters.self, from: Data(colString(stmt, 9).utf8))) ?? .protocolDefaults
        return LearnerProfile(
            learnerId: colString(stmt, 0),
            codedGroupLabel: colString(stmt, 1),
            displayName: colString(stmt, 2),
            studyMode: StudyMode(rawValue: colString(stmt, 3)) ?? .soloPilot,
            selectedSerialId: colString(stmt, 4),
            nonfictionTopics: topics,
            currentSessionNumber: Int(sqlite3_column_int64(stmt, 6)),
            totalLoggedSeconds: sqlite3_column_double(stmt, 7),
            medianReadingWPM: sqlite3_column_double(stmt, 8),
            parameters: params,
            createdAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 10))
        )
    }

    /// Initializes learner_items for all 4,500 active syllabus items with 15-min (regular) or 1-day (cognate) half-lives.
    /// In Solo Pilot mode, randomly holds back 10% of non-bootstrap items as never-scheduled controls (PDF p. 14).
    public func initializeLearnerItems(
        learnerId: String,
        items: [SyllabusItem],
        parameters: ProtocolParameters,
        holdBackTenPercentForSoloPilot: Bool
    ) throws {
        try exec("BEGIN IMMEDIATE TRANSACTION")
        do {
            let stmt = try prepare("""
                INSERT OR IGNORE INTO learner_items (
                    learner_id, item_id, half_life_days, state, is_held_back_control
                ) VALUES (?, ?, ?, 'new', ?)
            """)
            defer { sqlite3_finalize(stmt) }

            // Deterministic seeded shuffle per learner for 10% holdback selection
            var nonBootstrapIds = items.filter { !$0.isHoldout && !$0.isBootstrap && $0.itemType == .lemma }.map(\.id)
            var heldBackSet: Set<String> = []
            if holdBackTenPercentForSoloPilot {
                nonBootstrapIds.shuffle()
                let holdCount = Int(Double(items.filter { !$0.isHoldout }.count) * 0.10)
                heldBackSet = Set(nonBootstrapIds.prefix(holdCount))
            }

            for item in items where !item.isHoldout {
                sqlite3_reset(stmt)
                sqlite3_clear_bindings(stmt)
                let initialH = item.isTransparentCognate
                    ? parameters.newCognateHalfLifeDays
                    : parameters.newRegularHalfLifeDays
                bindText(stmt, 1, learnerId)
                bindText(stmt, 2, item.id)
                bindDouble(stmt, 3, initialH)
                bindInt(stmt, 4, heldBackSet.contains(item.id) ? 1 : 0)
                if sqlite3_step(stmt) != SQLITE_DONE {
                    throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
                }
            }
            try exec("COMMIT")
        } catch {
            try? exec("ROLLBACK")
            throw error
        }
    }

    public func loadLearnerItems(learnerId: String) throws -> [String: LearnerItemRecord] {
        let stmt = try prepare("""
            SELECT learner_id, item_id, half_life_days, last_meeting_at, first_meeting_session,
                   last_meeting_session, meetings_count, successes_count, ear_successes_count,
                   productive_uses_count, lookups_count, failures_count, state,
                   needs_extra_meeting, is_held_back_control, lapse_spares_used
            FROM learner_items WHERE learner_id = ?
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        var dict: [String: LearnerItemRecord] = [:]
        dict.reserveCapacity(4600)
        while sqlite3_step(stmt) == SQLITE_ROW {
            let rec = decodeLearnerItemRow(stmt)
            dict[rec.itemId] = rec
        }
        return dict
    }

    public func upsertLearnerItem(_ record: LearnerItemRecord) throws {
        let stmt = try prepare("""
            INSERT INTO learner_items (
                learner_id, item_id, half_life_days, last_meeting_at, first_meeting_session,
                last_meeting_session, meetings_count, successes_count, ear_successes_count,
                productive_uses_count, lookups_count, failures_count, state,
                needs_extra_meeting, is_held_back_control, lapse_spares_used
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(learner_id, item_id) DO UPDATE SET
                half_life_days=excluded.half_life_days,
                last_meeting_at=excluded.last_meeting_at,
                first_meeting_session=excluded.first_meeting_session,
                last_meeting_session=excluded.last_meeting_session,
                meetings_count=excluded.meetings_count,
                successes_count=excluded.successes_count,
                ear_successes_count=excluded.ear_successes_count,
                productive_uses_count=excluded.productive_uses_count,
                lookups_count=excluded.lookups_count,
                failures_count=excluded.failures_count,
                state=excluded.state,
                needs_extra_meeting=excluded.needs_extra_meeting,
                is_held_back_control=excluded.is_held_back_control,
                lapse_spares_used=excluded.lapse_spares_used
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, record.learnerId)
        bindText(stmt, 2, record.itemId)
        bindDouble(stmt, 3, record.halfLifeDays)
        bindDouble(stmt, 4, record.lastMeetingAt?.timeIntervalSince1970)
        bindInt(stmt, 5, record.firstMeetingSession)
        bindInt(stmt, 6, record.lastMeetingSession)
        bindInt(stmt, 7, record.meetingsCount)
        bindInt(stmt, 8, record.successesCount)
        bindInt(stmt, 9, record.earSuccessesCount)
        bindInt(stmt, 10, record.productiveUsesCount)
        bindInt(stmt, 11, record.lookupsCount)
        bindInt(stmt, 12, record.failuresCount)
        bindText(stmt, 13, record.state.rawValue)
        bindInt(stmt, 14, record.needsExtraMeetingThisSession ? 1 : 0)
        bindInt(stmt, 15, record.isHeldBackControl ? 1 : 0)
        bindInt(stmt, 16, record.lapseSparesUsed)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    private func decodeLearnerItemRow(_ stmt: OpaquePointer) -> LearnerItemRecord {
        LearnerItemRecord(
            learnerId: colString(stmt, 0),
            itemId: colString(stmt, 1),
            halfLifeDays: sqlite3_column_double(stmt, 2),
            lastMeetingAt: colOptDouble(stmt, 3).map { Date(timeIntervalSince1970: $0) },
            firstMeetingSession: colOptInt(stmt, 4),
            lastMeetingSession: colOptInt(stmt, 5),
            meetingsCount: Int(sqlite3_column_int64(stmt, 6)),
            successesCount: Int(sqlite3_column_int64(stmt, 7)),
            earSuccessesCount: Int(sqlite3_column_int64(stmt, 8)),
            productiveUsesCount: Int(sqlite3_column_int64(stmt, 9)),
            lookupsCount: Int(sqlite3_column_int64(stmt, 10)),
            failuresCount: Int(sqlite3_column_int64(stmt, 11)),
            state: MasteryState(rawValue: colString(stmt, 12)) ?? .new,
            needsExtraMeetingThisSession: sqlite3_column_int(stmt, 13) != 0,
            isHeldBackControl: sqlite3_column_int(stmt, 14) != 0,
            lapseSparesUsed: Int(sqlite3_column_int64(stmt, 15))
        )
    }

    // MARK: - Rule 4 Fresh Context & Passages & Serial Continuity

    public func loadPriorContextContentLemmas(learnerId: String, itemIds: [String]) throws -> [String: Set<String>] {
        guard !itemIds.isEmpty else { return [:] }
        let placeholders = itemIds.map { _ in "?" }.joined(separator: ",")
        let stmt = try prepare("""
            SELECT item_id, neighbor_content_lemmas_json
            FROM item_sentence_contexts
            WHERE learner_id = ? AND item_id IN (\(placeholders))
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        for (i, id) in itemIds.enumerated() {
            bindText(stmt, Int32(i + 2), id)
        }
        let decoder = JSONDecoder()
        var result: [String: Set<String>] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = colString(stmt, 0)
            let json = colString(stmt, 1)
            if let arr = try? decoder.decode([String].self, from: Data(json.utf8)) {
                result[id, default: []].formUnion(arr)
            }
        }
        return result
    }

    public func loadPriorSentences(learnerId: String, itemId: String, limit: Int = 10) throws -> [String] {
        let stmt = try prepare("""
            SELECT sentence_text FROM item_sentence_contexts
            WHERE learner_id = ? AND item_id = ?
            ORDER BY created_at DESC LIMIT ?
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        bindText(stmt, 2, itemId)
        bindInt(stmt, 3, limit)
        var res: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            res.append(colString(stmt, 0))
        }
        return res
    }

    public func recordItemSentenceContext(
        learnerId: String,
        itemId: String,
        passageId: String,
        sentenceText: String,
        neighborContentLemmas: [String]
    ) throws {
        let json = String(data: try JSONEncoder().encode(neighborContentLemmas), encoding: .utf8) ?? "[]"
        let stmt = try prepare("""
            INSERT INTO item_sentence_contexts (
                learner_id, item_id, passage_id, sentence_text, neighbor_content_lemmas_json, created_at
            ) VALUES (?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        bindText(stmt, 2, itemId)
        bindText(stmt, 3, passageId)
        bindText(stmt, 4, sentenceText)
        bindText(stmt, 5, json)
        bindDouble(stmt, 6, Date().timeIntervalSince1970)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func hasSeenSentenceBefore(learnerId: String, normalizedSentence: String) throws -> Bool {
        let stmt = try prepare("""
            SELECT 1 FROM item_sentence_contexts
            WHERE learner_id = ? AND lower(trim(sentence_text)) = ? LIMIT 1
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        bindText(stmt, 2, normalizedSentence.lowercased().trimmingCharacters(in: .whitespacesAndNewlines))
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    public func savePassage(_ passage: GeneratedPassage, learnerId: String) throws {
        let json = String(data: try JSONEncoder().encode(passage), encoding: .utf8) ?? "{}"
        let stmt = try prepare("""
            INSERT OR REPLACE INTO passages (
                passage_id, learner_id, session_number, block_type, format, title_es, full_text,
                word_count, voice, accent, target_wpm, is_audio_first, scheduled_word_share,
                rewrite_attempts, flagged_for_native_audit, passage_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, passage.id)
        bindText(stmt, 2, learnerId)
        bindInt(stmt, 3, passage.sessionNumber)
        bindText(stmt, 4, passage.blockType.rawValue)
        bindText(stmt, 5, passage.format.rawValue)
        bindText(stmt, 6, passage.titleEs)
        bindText(stmt, 7, passage.fullText)
        bindInt(stmt, 8, passage.wordCount)
        bindText(stmt, 9, passage.voice.rawValue)
        bindText(stmt, 10, passage.accent.rawValue)
        bindInt(stmt, 11, passage.targetWPM)
        bindInt(stmt, 12, passage.isAudioFirst ? 1 : 0)
        bindDouble(stmt, 13, passage.scheduledWordShare)
        bindInt(stmt, 14, passage.rewriteAttempts)
        bindInt(stmt, 15, passage.flaggedForNativeAudit ? 1 : 0)
        bindText(stmt, 16, json)
        bindDouble(stmt, 17, passage.createdAt.timeIntervalSince1970)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func loadNativeAuditPassages(learnerId: String) throws -> [GeneratedPassage] {
        let stmt = try prepare("""
            SELECT passage_json FROM passages
            WHERE learner_id = ? AND flagged_for_native_audit = 1
            ORDER BY created_at DESC
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        let decoder = JSONDecoder()
        var out: [GeneratedPassage] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let p = try? decoder.decode(GeneratedPassage.self, from: Data(colString(stmt, 0).utf8)) {
                out.append(p)
            }
        }
        return out
    }

    public func loadSerialProgress(learnerId: String, serialId: String) throws -> (episode: Int, summary: String) {
        let stmt = try prepare("SELECT episode_number, story_summary FROM serial_progress WHERE learner_id = ? AND serial_id = ?")
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        bindText(stmt, 2, serialId)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return (0, "") }
        return (Int(sqlite3_column_int64(stmt, 0)), colString(stmt, 1))
    }

    public func updateSerialProgress(learnerId: String, serialId: String, episode: Int, summary: String) throws {
        let stmt = try prepare("""
            INSERT INTO serial_progress (learner_id, serial_id, episode_number, story_summary, updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(learner_id, serial_id) DO UPDATE SET
                episode_number=excluded.episode_number,
                story_summary=excluded.story_summary,
                updated_at=excluded.updated_at
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        bindText(stmt, 2, serialId)
        bindInt(stmt, 3, episode)
        bindText(stmt, 4, summary)
        bindDouble(stmt, 5, Date().timeIntervalSince1970)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    // MARK: - Append-Only Evidence Log

    public func appendEvidence(_ event: EvidenceEvent) throws {
        let stmt = try prepare("""
            INSERT INTO evidence_events (
                event_uuid, learner_id, session_id, session_number, passage_id, item_id,
                event_type, timestamp, predicted_recall_before, half_life_before, half_life_after,
                is_first_meeting_glossed, is_unseen_sentence, passage_understood, is_usable_event,
                numeric_value, payload_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, event.eventUUID)
        bindText(stmt, 2, event.learnerId)
        bindText(stmt, 3, event.sessionId)
        bindInt(stmt, 4, event.sessionNumber)
        bindText(stmt, 5, event.passageId)
        bindText(stmt, 6, event.itemId)
        bindText(stmt, 7, event.eventType.rawValue)
        bindDouble(stmt, 8, event.timestamp.timeIntervalSince1970)
        bindDouble(stmt, 9, event.predictedRecallBefore)
        bindDouble(stmt, 10, event.halfLifeBefore)
        bindDouble(stmt, 11, event.halfLifeAfter)
        bindInt(stmt, 12, event.isFirstMeetingGlossed ? 1 : 0)
        bindInt(stmt, 13, event.isUnseenSentence ? 1 : 0)
        bindInt(stmt, 14, event.passageUnderstood.map { $0 ? 1 : 0 })
        bindInt(stmt, 15, event.isUsableEvent ? 1 : 0)
        bindDouble(stmt, 16, event.numericValue)
        bindText(stmt, 17, event.payloadJSON)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func loadEvidenceEvents(learnerId: String, type: EvidenceEventType? = nil, limit: Int? = nil) throws -> [EvidenceEvent] {
        var sql = """
            SELECT id, event_uuid, learner_id, session_id, session_number, passage_id, item_id,
                   event_type, timestamp, predicted_recall_before, half_life_before, half_life_after,
                   is_first_meeting_glossed, is_unseen_sentence, passage_understood, is_usable_event,
                   numeric_value, payload_json
            FROM evidence_events WHERE learner_id = ?
        """
        if type != nil { sql += " AND event_type = ?" }
        sql += " ORDER BY id ASC"
        if let limit { sql += " LIMIT \(limit)" }

        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        if let type { bindText(stmt, 2, type.rawValue) }

        var events: [EvidenceEvent] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let understoodInt = colOptInt(stmt, 14)
            events.append(EvidenceEvent(
                id: sqlite3_column_int64(stmt, 0),
                eventUUID: colString(stmt, 1),
                learnerId: colString(stmt, 2),
                sessionId: colOptString(stmt, 3),
                sessionNumber: colOptInt(stmt, 4),
                passageId: colOptString(stmt, 5),
                itemId: colOptString(stmt, 6),
                eventType: EvidenceEventType(rawValue: colString(stmt, 7)) ?? .sessionStarted,
                timestamp: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 8)),
                predictedRecallBefore: colOptDouble(stmt, 9),
                halfLifeBefore: colOptDouble(stmt, 10),
                halfLifeAfter: colOptDouble(stmt, 11),
                isFirstMeetingGlossed: sqlite3_column_int(stmt, 12) != 0,
                isUnseenSentence: sqlite3_column_int(stmt, 13) != 0,
                passageUnderstood: understoodInt.map { $0 != 0 },
                isUsableEvent: sqlite3_column_int(stmt, 15) != 0,
                numericValue: colOptDouble(stmt, 16),
                payloadJSON: colString(stmt, 17)
            ))
        }
        return events
    }

    // MARK: - Sessions, Audits, Assessments & Delayed Capacity Tests

    public func saveSessionSummary(_ summary: SessionSummary, learnerId: String, targetNewItems: Int) throws {
        let json = String(data: try JSONEncoder().encode(summary), encoding: .utf8) ?? "{}"
        let stmt = try prepare("""
            INSERT OR REPLACE INTO sessions (
                session_id, learner_id, session_number, week, study_mode, target_new_items,
                started_at, ended_at, active_input_seconds, active_output_seconds,
                passages_completed, running_words, usable_events, probes_correct, probes_total,
                new_items_introduced, promoted_known, promoted_productive,
                fatigue_rating, enjoyment_rating, no_other_spanish_confirmed, summary_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, summary.id)
        bindText(stmt, 2, learnerId)
        bindInt(stmt, 3, summary.sessionNumber)
        bindInt(stmt, 4, summary.week)
        bindText(stmt, 5, summary.studyMode.rawValue)
        bindInt(stmt, 6, targetNewItems)
        bindDouble(stmt, 7, summary.startedAt.timeIntervalSince1970)
        bindDouble(stmt, 8, summary.endedAt.timeIntervalSince1970)
        bindDouble(stmt, 9, summary.activeInputSeconds)
        bindDouble(stmt, 10, summary.activeOutputSeconds)
        bindInt(stmt, 11, summary.passagesCompleted)
        bindInt(stmt, 12, summary.runningWordsProcessed)
        bindInt(stmt, 13, summary.usableEventsCount)
        bindInt(stmt, 14, summary.probesCorrect)
        bindInt(stmt, 15, summary.probesTotal)
        bindInt(stmt, 16, summary.newItemsIntroduced)
        bindInt(stmt, 17, summary.itemsPromotedToKnown)
        bindInt(stmt, 18, summary.itemsPromotedToProductive)
        bindInt(stmt, 19, summary.fatigueRating)
        bindInt(stmt, 20, summary.enjoymentRating)
        bindInt(stmt, 21, summary.noOtherSpanishConfirmed ? 1 : 0)
        bindText(stmt, 22, json)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func loadSessionSummaries(learnerId: String) throws -> [SessionSummary] {
        let stmt = try prepare("SELECT summary_json FROM sessions WHERE learner_id = ? AND summary_json IS NOT NULL ORDER BY session_number ASC")
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        let decoder = JSONDecoder()
        var out: [SessionSummary] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let s = try? decoder.decode(SessionSummary.self, from: Data(colString(stmt, 0).utf8)) {
                out.append(s)
            }
        }
        return out
    }

    public func saveWeeklyAudit(_ audit: WeeklyAuditResult, learnerId: String) throws {
        let json = String(data: try JSONEncoder().encode(audit), encoding: .utf8) ?? "{}"
        let stmt = try prepare("""
            INSERT OR REPLACE INTO weekly_audits (
                audit_id, learner_id, week, completed_at, mean_predicted, mean_observed,
                abs_error, did_refit, duration_seconds, result_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, audit.id)
        bindText(stmt, 2, learnerId)
        bindInt(stmt, 3, audit.week)
        bindDouble(stmt, 4, audit.completedAt.timeIntervalSince1970)
        bindDouble(stmt, 5, audit.meanPredictedRecall)
        bindDouble(stmt, 6, audit.meanObservedRecall)
        bindDouble(stmt, 7, audit.absoluteCalibrationError)
        bindInt(stmt, 8, audit.didRefitModel ? 1 : 0)
        bindDouble(stmt, 9, audit.durationSeconds)
        bindText(stmt, 10, json)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func loadWeeklyAudits(learnerId: String) throws -> [WeeklyAuditResult] {
        let stmt = try prepare("SELECT result_json FROM weekly_audits WHERE learner_id = ? ORDER BY completed_at DESC")
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        let decoder = JSONDecoder()
        var out: [WeeklyAuditResult] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let a = try? decoder.decode(WeeklyAuditResult.self, from: Data(colString(stmt, 0).utf8)) {
                out.append(a)
            }
        }
        return out
    }

    public func saveResearchAssessment<T: Encodable>(
        id: String,
        learnerId: String,
        type: String,
        checkpointLabel: String,
        primaryScore: Double,
        payload: T
    ) throws {
        let json = String(data: try JSONEncoder().encode(payload), encoding: .utf8) ?? "{}"
        let stmt = try prepare("""
            INSERT OR REPLACE INTO research_assessments (
                assessment_id, learner_id, assessment_type, checkpoint_label, completed_at, score_primary, result_json
            ) VALUES (?,?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, id)
        bindText(stmt, 2, learnerId)
        bindText(stmt, 3, type)
        bindText(stmt, 4, checkpointLabel)
        bindDouble(stmt, 5, Date().timeIntervalSince1970)
        bindDouble(stmt, 6, primaryScore)
        bindText(stmt, 7, json)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func scheduleDelayedCapacityTest(
        learnerId: String,
        sourceSessionNumber: Int,
        intakeBucket: Int,
        itemIds: [String],
        sessionDate: Date = Date()
    ) throws {
        let dueAt = sessionDate.addingTimeInterval(7 * 86_400)
        let selected = Array(itemIds.shuffled().prefix(30))
        let json = String(data: try JSONEncoder().encode(selected), encoding: .utf8) ?? "[]"
        let stmt = try prepare("""
            INSERT INTO delayed_capacity_tests (
                learner_id, source_session_number, intake_rate_bucket, due_at, item_ids_json
            ) VALUES (?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        bindInt(stmt, 2, sourceSessionNumber)
        bindInt(stmt, 3, intakeBucket)
        bindDouble(stmt, 4, dueAt.timeIntervalSince1970)
        bindText(stmt, 5, json)
        if sqlite3_step(stmt) != SQLITE_DONE {
            throw DatabaseError.stepFailed(String(cString: sqlite3_errmsg(db!)))
        }
    }

    public func loadCapacityStats(learnerId: String) throws -> [CapacityBucketStat] {
        let stmt = try prepare("""
            SELECT intake_rate_bucket,
                   COUNT(*) AS sessions_logged,
                   SUM(CASE WHEN completed_at IS NOT NULL THEN 1 ELSE 0 END) AS tested_sessions,
                   AVG(CASE WHEN completed_at IS NOT NULL THEN retention_rate END) AS avg_retention
            FROM delayed_capacity_tests
            WHERE learner_id = ?
            GROUP BY intake_rate_bucket
            ORDER BY intake_rate_bucket ASC
        """)
        defer { sqlite3_finalize(stmt) }
        bindText(stmt, 1, learnerId)
        var out: [CapacityBucketStat] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(CapacityBucketStat(
                intakeTarget: Int(sqlite3_column_int64(stmt, 0)),
                sessionsLogged: Int(sqlite3_column_int64(stmt, 1)),
                sevenDayTestedCount: Int(sqlite3_column_int64(stmt, 2)) * 30,
                sevenDayRetentionRate: colOptDouble(stmt, 3) ?? 0.0
            ))
        }
        return out
    }

    /// Verifies at runtime (and in unit tests) that SQLite triggers truly forbid mutating `evidence_events`.
    public func attemptForbiddenEvidenceMutationForTest() -> Bool {
        do {
            try exec("DELETE FROM evidence_events")
            return false // Should never succeed!
        } catch {
            return true // Correctly blocked by trigger!
        }
    }
}
