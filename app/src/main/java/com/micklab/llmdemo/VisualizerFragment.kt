package com.micklab.llmdemo

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.core.widget.addTextChangedListener
import androidx.fragment.app.Fragment
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

class VisualizerFragment : Fragment() {
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val generation = AtomicInteger(0)

    private var modelBundle: ModelBundle? = null
    private var inputEditor: EditText? = null
    private var runButton: Button? = null
    private var statusText: TextView? = null
    private var heatmapView: HeatmapView? = null
    private var embeddingPlotView: EmbeddingPlotView? = null
    private var logitsChart: LogitsChart? = null

    override fun onCreateView(
        inflater: android.view.LayoutInflater,
        container: android.view.ViewGroup?,
        savedInstanceState: Bundle?,
    ): View {
        val context = requireContext()
        val scrollView = ScrollView(context).apply {
            isFillViewport = true
        }

        val content = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(24))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        val title = TextView(context).apply {
            text = "Mini Transformer Visualizer"
            textSize = 24f
        }
        val subtitle = TextView(context).apply {
            text =
                "Type a short lowercase prompt to inspect token embeddings, causal attention, and next-token logits in real time."
            textSize = 14f
            setPadding(0, dp(8), 0, dp(16))
        }
        inputEditor = EditText(context).apply {
            hint = "Example: attention on android"
            setText("attention on android")
            minLines = 3
            gravity = Gravity.TOP
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            isEnabled = false
        }
        runButton = Button(context).apply {
            text = "Run forward()"
            isEnabled = false
            setOnClickListener {
                scheduleInference(inputEditor?.text?.toString().orEmpty(), force = true)
            }
        }
        statusText = TextView(context).apply {
            text = "Loading model assets…"
            textSize = 14f
            setPadding(0, dp(12), 0, dp(16))
        }
        heatmapView = HeatmapView(context).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(280))
        }
        embeddingPlotView = EmbeddingPlotView(context).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(280))
        }
        logitsChart = LogitsChart(context).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(300))
        }

        content.addView(title)
        content.addView(subtitle)
        content.addView(inputEditor)
        content.addView(runButton)
        content.addView(statusText)
        content.addView(sectionTitle("Attention Heatmap"))
        content.addView(sectionDescription("Each cell shows how strongly a token attends to earlier tokens after the causal mask is applied."))
        content.addView(heatmapView)
        content.addView(sectionTitle("Embedding PCA"))
        content.addView(sectionDescription("Token embeddings are projected from 32 dimensions down to 2D with PCA so geometry is easy to inspect."))
        content.addView(embeddingPlotView)
        content.addView(sectionTitle("Logits Bar Chart"))
        content.addView(sectionDescription("The chart shows the largest next-token logits from the last position in the sequence."))
        content.addView(logitsChart)

        scrollView.addView(content)

        inputEditor?.addTextChangedListener { editable ->
            scheduleInference(editable?.toString().orEmpty())
        }

        return scrollView
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        loadModel()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        inputEditor = null
        runButton = null
        statusText = null
        heatmapView = null
        embeddingPlotView = null
        logitsChart = null
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdownNow()
    }

    private fun loadModel() {
        val appContext = requireContext().applicationContext
        statusText?.text = "Loading weights and vocabulary from app assets…"

        executor.execute {
            runCatching {
                MiniTransformer.loadBundle(appContext)
            }.onSuccess { bundle ->
                mainHandler.post {
                    if (!isAdded) return@post
                    modelBundle = bundle
                    inputEditor?.isEnabled = true
                    runButton?.isEnabled = true
                    statusText?.text =
                        "Model ready: vocab=${bundle.model.config.vocabSize}, d_model=${bundle.model.config.dModel}, layers=${bundle.model.config.numLayers}."
                    scheduleInference(inputEditor?.text?.toString().orEmpty(), force = true)
                }
            }.onFailure { error ->
                mainHandler.post {
                    if (!isAdded) return@post
                    statusText?.text = "Model load failed: ${error.message}"
                }
            }
        }
    }

    private fun scheduleInference(prompt: String, force: Boolean = false) {
        val bundle = modelBundle ?: return
        if (!force && prompt.isBlank()) return

        val requestId = generation.incrementAndGet()
        val sanitizedPrompt = prompt.ifBlank { "attention on android" }
        statusText?.text = "Running forward() for ${sanitizedPrompt.length} characters…"

        executor.execute {
            runCatching {
                val snapshots = linkedMapOf<String, TensorSnapshot>()
                val tokenIds = bundle.tokenizer.encode(sanitizedPrompt, bundle.model.config.maxSeqLen)
                val result = bundle.model.forward(tokenIds) { snapshot ->
                    snapshots[snapshot.name] = snapshot
                }

                val tokenLabels = bundle.tokenizer.displayLabels(result.tokenIds)
                val attentionMatrix =
                    (snapshots["layer_0_attention_weights"] as? TensorSnapshot.MatrixSnapshot)?.values
                        ?: emptyArray()
                val embeddingMatrix =
                    (snapshots["embedding_plus_position"] as? TensorSnapshot.MatrixSnapshot)?.values
                        ?: result.finalHiddenStates
                val topIndices = TensorUtils.topK(result.logits, 12)
                val bars = topIndices.map { tokenId ->
                    LogitBar(
                        label = bundle.tokenizer.displayLabel(tokenId),
                        value = result.logits[tokenId],
                    )
                }
                val predictedToken = bundle.tokenizer.displayLabel(TensorUtils.argmax(result.logits))
                val readablePrompt = bundle.tokenizer.decode(result.tokenIds)

                Triple(
                    """
                    Prompt: "$readablePrompt"
                    Tokens: ${tokenLabels.joinToString(" ")}
                    Predicted next char: $predictedToken
                    Sequence length: ${result.tokenIds.size} / ${bundle.model.config.maxSeqLen}
                    """.trimIndent(),
                    Triple(attentionMatrix, embeddingMatrix, tokenLabels),
                    bars,
                )
            }.onSuccess { (status, matrices, bars) ->
                mainHandler.post {
                    if (!isAdded || requestId != generation.get()) return@post
                    val (attention, embeddings, tokenLabels) = matrices
                    statusText?.text = status
                    heatmapView?.submit(attention, tokenLabels)
                    embeddingPlotView?.submitEmbeddings(embeddings, tokenLabels)
                    logitsChart?.submit(bars)
                }
            }.onFailure { error ->
                mainHandler.post {
                    if (!isAdded || requestId != generation.get()) return@post
                    statusText?.text = "Inference failed: ${error.message}"
                }
            }
        }
    }

    private fun sectionTitle(text: String): TextView =
        TextView(requireContext()).apply {
            setPadding(0, dp(20), 0, dp(4))
            textSize = 18f
            this.text = text
        }

    private fun sectionDescription(text: String): TextView =
        TextView(requireContext()).apply {
            setPadding(0, 0, 0, dp(8))
            textSize = 13f
            this.text = text
        }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
