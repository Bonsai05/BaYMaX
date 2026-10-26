
package com.baymax.android.core.storage

import android.content.Context
import android.database.sqlite.SQLiteConstraintException
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BaymaxStorageInstrumentedTest {

    private lateinit var context: Context
    private lateinit var databaseName: String
    private lateinit var database: BaymaxDatabase
    private lateinit var repository: EventRepository

    @Before
    fun setUp() {
        context = InstrumentationRegistry
            .getInstrumentation()
            .targetContext

        databaseName = "test_${System.nanoTime()}.db"
        database = BaymaxDatabase(context, databaseName)
        repository = EventRepository(database, PayloadCipher())
    }

    @After
    fun tearDown() {
        database.close()
        context.deleteDatabase(databaseName)
    }

    @Test
    fun eventIsPersistedAndReadableAfterDatabaseReopen() {
        val created = repository.createEvent("BAYMAX persistence test")

        database.close()

        database = BaymaxDatabase(context, databaseName)
        repository = EventRepository(database, PayloadCipher())

        val restored = repository.getEvents()
            .single { it.id == created.id }

        assertEquals("BAYMAX persistence test", restored.text)
        assertEquals("PENDING", restored.syncStatus)
    }

    @Test
    fun payloadIsEncryptedInDatabase() {
        val secret = "private local note"
        repository.createEvent(secret)

        val stored = database.getAllEvents().single()

        assertFalse(
            stored.payload.ciphertext.contentEquals(
                secret.toByteArray(Charsets.UTF_8)
            )
        )

        assertEquals(
            secret,
            PayloadCipher().decrypt(stored.payload)
        )
    }

    @Test
    fun retryMetadataPersists() {
        val event = repository.createEvent("Retry me")

        assertTrue(
            repository.recordRetry(
                eventId = event.id,
                attempts = 1,
                error = "Core unavailable"
            )
        )

        val restored = database.getAllEvents()
            .single { it.id == event.id }

        assertEquals(1, restored.attempts)
        assertEquals("Core unavailable", restored.lastError)
        assertTrue(restored.nextRetryAt > System.currentTimeMillis())
        assertEquals("PENDING", restored.syncStatus)
    }

    @Test
    fun duplicateEventIdIsRejected() {
        repository.createEvent("Unique event")

        val stored = database.getAllEvents().single()

        try {
            database.insertEvent(stored)
            fail("Expected duplicate event ID to be rejected")
        } catch (_: SQLiteConstraintException) {
            // Expected: the event primary key prevents duplicates.
        }
    }

    @Test
    fun duePendingQueryRespectsRetryTime() {
        val event = repository.createEvent("Wait before retry")

        repository.recordRetry(
            eventId = event.id,
            attempts = 1,
            error = "Temporary failure"
        )

        val now = System.currentTimeMillis()

        assertTrue(
            repository.getDuePendingEvents(now).none {
                it.id == event.id
            }
        )

        assertTrue(
            repository.getDuePendingEvents(now + 120_000).any {
                it.id == event.id
            }
        )
    }
}
