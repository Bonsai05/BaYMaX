
package com.baymax.android.core.storage

import java.util.UUID
import kotlin.math.min

data class EventItem(
    val id: String,
    val text: String,
    val createdAt: Long,
    val syncStatus: String,
    val attempts: Int = 0,
    val nextRetryAt: Long = 0L,
    val lastError: String? = null
)

class EventRepository(
    private val database: BaymaxDatabase,
    private val cipher: PayloadCipher
) {

    fun createEvent(text: String): EventItem {
        require(text.isNotBlank()) { "Event text cannot be empty" }

        val cleanText = text.trim()
        val id = UUID.randomUUID().toString()
        val now = System.currentTimeMillis()

        database.insertEvent(
            StoredEvent(
                id = id,
                payload = cipher.encrypt(cleanText),
                createdAt = now,
                syncStatus = "PENDING"
            )
        )

        return EventItem(id, cleanText, now, "PENDING")
    }

    fun getEvents(): List<EventItem> =
        database.getAllEvents().map { it.toEventItem() }

    fun getDuePendingEvents(
        now: Long = System.currentTimeMillis(),
        limit: Int = 50
    ): List<EventItem> =
        database.getDuePendingEvents(now, limit)
            .map { it.toEventItem() }

    fun markSynced(eventId: String): Boolean =
        database.updateSyncStatus(eventId, "SYNCED")

    fun recordRetry(
        eventId: String,
        attempts: Int,
        error: String,
        now: Long = System.currentTimeMillis()
    ): Boolean {
        val exponent = attempts.coerceIn(0, 10)
        val delaySeconds = min(3600L, 30L * (1L shl exponent))

        return database.recordRetry(
            eventId = eventId,
            attempts = attempts,
            nextRetryAt = now + delaySeconds * 1000L,
            error = error
        )
    }

    private fun StoredEvent.toEventItem() =
        EventItem(
            id = id,
            text = cipher.decrypt(payload),
            createdAt = createdAt,
            syncStatus = syncStatus,
            attempts = attempts,
            nextRetryAt = nextRetryAt,
            lastError = lastError
        )
}
