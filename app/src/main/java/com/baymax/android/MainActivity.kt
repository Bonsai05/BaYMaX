
package com.baymax.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.baymax.android.core.storage.BaymaxDatabase
import com.baymax.android.core.storage.EventItem
import com.baymax.android.core.storage.EventRepository
import com.baymax.android.core.storage.PayloadCipher
import com.baymax.android.core.sync.OutboxScheduler
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private val executor = Executors.newSingleThreadExecutor()

    private lateinit var repository: EventRepository

    private var events by mutableStateOf<List<EventItem>>(emptyList())
    private var loading by mutableStateOf(true)
    private var error by mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        repository = EventRepository(
            BaymaxDatabase(applicationContext),
            PayloadCipher()
        )

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    HomeScreen(
                        events = events,
                        loading = loading,
                        error = error,
                        onSave = ::saveEvent,
                        onRefresh = ::refreshEvents
                    )
                }
            }
        }

        refreshEvents()
    }

    private fun refreshEvents() {
        loading = true

        executor.execute {
            try {
                val result = repository.getEvents()

                runOnUiThread {
                    events = result
                    error = null
                    loading = false
                }
            } catch (e: Exception) {
                runOnUiThread {
                    error = e.message ?: "Could not load local events"
                    loading = false
                }
            }
        }
    }

    private fun saveEvent(text: String) {
        if (text.isBlank()) return

        loading = true

        executor.execute {
            try {
                // 1. Save the encrypted event in SQLite.
                repository.createEvent(text)

                // 2. Schedule the durable outbox worker.
                OutboxScheduler.enqueue(applicationContext)

                // 3. Refresh the local event list.
                val result = repository.getEvents()

                runOnUiThread {
                    events = result
                    error = null
                    loading = false
                }
            } catch (e: Exception) {
                runOnUiThread {
                    error = e.message ?: "Could not save event"
                    loading = false
                }
            }
        }
    }

    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }
}

@Composable
private fun HomeScreen(
    events: List<EventItem>,
    loading: Boolean,
    error: String?,
    onSave: (String) -> Unit,
    onRefresh: () -> Unit
) {
    var input by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "BAYMAX",
            style = MaterialTheme.typography.headlineLarge
        )

        Text("Local-first device node")

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text("Storage: SQLite")
                Text("Core connection: Not configured")
                Text("Pending: ${events.count { it.syncStatus == "PENDING" }}")
            }
        }

        OutlinedTextField(
            value = input,
            onValueChange = { input = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Event text") },
            placeholder = { Text("Enter a task or note") },
            enabled = !loading
        )

        Button(
            onClick = {
                val text = input.trim()
                if (text.isNotEmpty()) {
                    onSave(text)
                    input = ""
                }
            },
            enabled = input.isNotBlank() && !loading,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Save locally")
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "Local events",
                style = MaterialTheme.typography.titleLarge
            )

            Button(
                onClick = onRefresh,
                enabled = !loading
            ) {
                Text("Refresh")
            }
        }

        if (loading) {
            Text("Loading...")
        }

        error?.let {
            Text(
                text = "Error: $it",
                color = MaterialTheme.colorScheme.error
            )
        }

        LazyColumn(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(events, key = { it.id }) { event ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(event.text)
                        Text("Status: ${event.syncStatus}")
                        Text(
                            text = event.id,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }
            }
        }
    }
}
