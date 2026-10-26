
package com.baymax.android.core.sync

import com.baymax.android.core.storage.EventItem

sealed interface SyncResult {
    data object Acknowledged : SyncResult
    data class RetryableFailure(val reason: String) : SyncResult
    data class PermanentFailure(val reason: String) : SyncResult
}

interface SyncTransport {
    /**
     * Return Acknowledged only after the Core confirms durable commit.
     * Implementations must use event.id as the idempotency key.
     */
    fun send(event: EventItem): SyncResult
}

object SyncTransportRegistry {
    @Volatile
    var transport: SyncTransport? = null
}
