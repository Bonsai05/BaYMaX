
package com.baymax.android.core.sync

import android.content.Context
import androidx.work.Worker
import androidx.work.WorkerParameters
import com.baymax.android.core.storage.BaymaxDatabase
import com.baymax.android.core.storage.EventRepository
import com.baymax.android.core.storage.PayloadCipher

class OutboxSyncWorker(
    context: Context,
    params: WorkerParameters
) : Worker(context, params) {

    override fun doWork(): Result {
        val transport = SyncTransportRegistry.transport
            ?: return Result.failure()

        val repository = EventRepository(
            BaymaxDatabase(applicationContext),
            PayloadCipher()
        )

        return try {
            val pending = repository.getDuePendingEvents()

            if (pending.isEmpty()) {
                // Do not discard events that are waiting for their retry time.
                val hasPending = repository.getEvents()
                    .any { it.syncStatus == "PENDING" }

                return if (hasPending) Result.retry() else Result.success()
            }

            for (event in pending) {
                when (val result = transport.send(event)) {
                    SyncResult.Acknowledged -> {
                        if (!repository.markSynced(event.id)) {
                            return Result.retry()
                        }
                    }

                    is SyncResult.RetryableFailure -> {
                        repository.recordRetry(
                            eventId = event.id,
                            attempts = event.attempts + 1,
                            error = result.reason
                        )
                        return Result.retry()
                    }

                    is SyncResult.PermanentFailure -> {
                        // Preserve the event for inspection and manual recovery.
                        repository.recordRetry(
                            eventId = event.id,
                            attempts = event.attempts + 1,
                            error = "Permanent: ${result.reason}"
                        )
                        return Result.failure()
                    }
                }
            }

            // Continue processing additional batches if necessary.
            if (repository.getEvents().any { it.syncStatus == "PENDING" }) {
                Result.retry()
            } else {
                Result.success()
            }
        } catch (_: Exception) {
            Result.retry()
        }
    }
}
