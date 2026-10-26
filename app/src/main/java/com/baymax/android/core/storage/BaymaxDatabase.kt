
package com.baymax.android.core.storage

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.util.UUID

data class StoredEvent(
    val id: String,
    val payload: EncryptedPayload,
    val createdAt: Long,
    val syncStatus: String,
    val attempts: Int = 0,
    val nextRetryAt: Long = 0L,
    val lastError: String? = null
)

class BaymaxDatabase(
    context: Context,
    databaseName: String = "baymax_local.db"
) : SQLiteOpenHelper(
    context.applicationContext,
    databaseName,
    null,
    2
) {

    companion object {
        private const val TABLE = "local_events"
        private const val ID = "event_id"
        private const val PAYLOAD = "payload"
        private const val IV = "iv"
        private const val CREATED = "created_at"
        private const val STATUS = "sync_status"
        private const val ATTEMPTS = "attempt_count"
        private const val NEXT_RETRY = "next_retry_at"
        private const val LAST_ERROR = "last_error"
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE $TABLE (
                $ID TEXT PRIMARY KEY,
                $PAYLOAD BLOB NOT NULL,
                $IV BLOB NOT NULL,
                $CREATED INTEGER NOT NULL,
                $STATUS TEXT NOT NULL
                    CHECK ($STATUS IN ('PENDING', 'SYNCED', 'FAILED')),
                $ATTEMPTS INTEGER NOT NULL DEFAULT 0,
                $NEXT_RETRY INTEGER NOT NULL DEFAULT 0,
                $LAST_ERROR TEXT
            )
            """.trimIndent()
        )

        createIndexes(db)
    }

    private fun createIndexes(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE INDEX IF NOT EXISTS idx_events_status_retry
            ON $TABLE ($STATUS, $NEXT_RETRY, $CREATED)
            """.trimIndent()
        )
    }

    override fun onUpgrade(
        db: SQLiteDatabase,
        oldVersion: Int,
        newVersion: Int
    ) {
        if (oldVersion < 2) {
            db.execSQL(
                "ALTER TABLE $TABLE ADD COLUMN $ATTEMPTS INTEGER NOT NULL DEFAULT 0"
            )
            db.execSQL(
                "ALTER TABLE $TABLE ADD COLUMN $NEXT_RETRY INTEGER NOT NULL DEFAULT 0"
            )
            db.execSQL(
                "ALTER TABLE $TABLE ADD COLUMN $LAST_ERROR TEXT"
            )
            createIndexes(db)
        }
    }

    fun insertEvent(event: StoredEvent) {
        val values = ContentValues().apply {
            put(ID, event.id)
            put(PAYLOAD, event.payload.ciphertext)
            put(IV, event.payload.iv)
            put(CREATED, event.createdAt)
            put(STATUS, event.syncStatus)
            put(ATTEMPTS, event.attempts)
            put(NEXT_RETRY, event.nextRetryAt)
            put(LAST_ERROR, event.lastError)
        }

        writableDatabase.insertOrThrow(TABLE, null, values)
    }

    fun getAllEvents(): List<StoredEvent> {
        val events = mutableListOf<StoredEvent>()

        readableDatabase.query(
            TABLE,
            arrayOf(
                ID, PAYLOAD, IV, CREATED, STATUS,
                ATTEMPTS, NEXT_RETRY, LAST_ERROR
            ),
            null, null, null, null,
            "$CREATED DESC"
        ).use { cursor ->
            while (cursor.moveToNext()) {
                events += cursor.toStoredEvent()
            }
        }

        return events
    }

    fun getDuePendingEvents(
        now: Long,
        limit: Int = 50
    ): List<StoredEvent> {
        require(limit > 0)

        val events = mutableListOf<StoredEvent>()

        readableDatabase.query(
            TABLE,
            arrayOf(
                ID, PAYLOAD, IV, CREATED, STATUS,
                ATTEMPTS, NEXT_RETRY, LAST_ERROR
            ),
            "$STATUS = ? AND $NEXT_RETRY <= ?",
            arrayOf("PENDING", now.toString()),
            null,
            null,
            "$CREATED ASC",
            limit.toString()
        ).use { cursor ->
            while (cursor.moveToNext()) {
                events += cursor.toStoredEvent()
            }
        }

        return events
    }

    private fun android.database.Cursor.toStoredEvent() =
        StoredEvent(
            id = getString(0),
            payload = EncryptedPayload(
                ciphertext = getBlob(1),
                iv = getBlob(2)
            ),
            createdAt = getLong(3),
            syncStatus = getString(4),
            attempts = getInt(5),
            nextRetryAt = getLong(6),
            lastError = if (isNull(7)) null else getString(7)
        )

    fun updateSyncStatus(
        eventId: String,
        status: String
    ): Boolean {
        require(status in setOf("PENDING", "SYNCED", "FAILED"))

        val values = ContentValues().apply {
            put(STATUS, status)
            if (status == "SYNCED") {
                put(LAST_ERROR, null as String?)
            }
        }

        return writableDatabase.update(
            TABLE, values, "$ID = ?", arrayOf(eventId)
        ) == 1
    }

    fun recordRetry(
        eventId: String,
        attempts: Int,
        nextRetryAt: Long,
        error: String
    ): Boolean {
        val values = ContentValues().apply {
            put(ATTEMPTS, attempts)
            put(NEXT_RETRY, nextRetryAt)
            put(LAST_ERROR, error.take(500))
        }

        return writableDatabase.update(
            TABLE, values, "$ID = ?", arrayOf(eventId)
        ) == 1
    }

    fun newEventId(): String = UUID.randomUUID().toString()
}
