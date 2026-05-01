package com.micklab.llmdemo

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class ModelConfig(
    val vocabSize: Int,
    val dModel: Int,
    val maxSeqLen: Int,
    val numLayers: Int,
    val ffHiddenDim: Int,
)

data class TransformerLayerWeights(
    val wQ: Matrix,
    val wK: Matrix,
    val wV: Matrix,
    val wO: Matrix,
    val w1: Matrix,
    val w2: Matrix,
    val norm1Gamma: FloatArray,
    val norm1Beta: FloatArray,
    val norm2Gamma: FloatArray,
    val norm2Beta: FloatArray,
)

data class TransformerWeights(
    val config: ModelConfig,
    val embedding: Matrix,
    val layers: List<TransformerLayerWeights>,
) {
    companion object {
        fun fromJson(rawJson: String): TransformerWeights = fromJson(JSONObject(rawJson))

        fun fromJson(json: JSONObject): TransformerWeights {
            val configJson = json.getJSONObject("config")
            val config = ModelConfig(
                vocabSize = configJson.getInt("vocab_size"),
                dModel = configJson.getInt("d_model"),
                maxSeqLen = configJson.getInt("max_seq_len"),
                numLayers = configJson.getInt("num_layers"),
                ffHiddenDim = configJson.getInt("ff_hidden_dim"),
            )

            val embedding = jsonArrayToMatrix(json.getJSONArray("embedding"))
            require(embedding.size == config.vocabSize) {
                "Embedding table row count ${embedding.size} does not match vocab size ${config.vocabSize}."
            }
            require(embedding.firstOrNull()?.size == config.dModel) {
                "Embedding dimension does not match d_model ${config.dModel}."
            }

            val layersJson = json.getJSONArray("layers")
            require(layersJson.length() == config.numLayers) {
                "Weight file contains ${layersJson.length()} layers, expected ${config.numLayers}."
            }

            val layers = buildList(layersJson.length()) {
                for (index in 0 until layersJson.length()) {
                    val layer = layersJson.getJSONObject(index)
                    add(
                        TransformerLayerWeights(
                            wQ = jsonArrayToMatrix(layer.getJSONArray("w_q")),
                            wK = jsonArrayToMatrix(layer.getJSONArray("w_k")),
                            wV = jsonArrayToMatrix(layer.getJSONArray("w_v")),
                            wO = jsonArrayToMatrix(layer.getJSONArray("w_o")),
                            w1 = jsonArrayToMatrix(layer.getJSONArray("w1")),
                            w2 = jsonArrayToMatrix(layer.getJSONArray("w2")),
                            norm1Gamma = jsonArrayToFloatArray(layer.getJSONArray("norm1_gamma")),
                            norm1Beta = jsonArrayToFloatArray(layer.getJSONArray("norm1_beta")),
                            norm2Gamma = jsonArrayToFloatArray(layer.getJSONArray("norm2_gamma")),
                            norm2Beta = jsonArrayToFloatArray(layer.getJSONArray("norm2_beta")),
                        ),
                    )
                }
            }

            return TransformerWeights(config = config, embedding = embedding, layers = layers)
        }

        private fun jsonArrayToMatrix(jsonArray: JSONArray): Matrix =
            Array(jsonArray.length()) { row ->
                jsonArrayToFloatArray(jsonArray.getJSONArray(row))
            }

        private fun jsonArrayToFloatArray(jsonArray: JSONArray): FloatArray =
            FloatArray(jsonArray.length()) { index ->
                jsonArray.getDouble(index).toFloat()
            }
    }
}

sealed class TensorSnapshot(open val name: String) {
    data class MatrixSnapshot(
        override val name: String,
        val values: Matrix,
        val rowLabels: List<String> = emptyList(),
        val columnLabels: List<String> = emptyList(),
    ) : TensorSnapshot(name)

    data class VectorSnapshot(
        override val name: String,
        val values: FloatArray,
        val labels: List<String> = emptyList(),
    ) : TensorSnapshot(name)
}

data class ForwardResult(
    val tokenIds: IntArray,
    val finalHiddenStates: Matrix,
    val logits: FloatArray,
)

data class ModelBundle(
    val tokenizer: CharTokenizer,
    val model: MiniTransformer,
)

class CharTokenizer private constructor(private val vocabulary: List<String>) {
    private val tokenToId = vocabulary.withIndex().associate { (index, token) -> token to index }

    val size: Int
        get() = vocabulary.size

    private val padId = tokenToId.getValue("[PAD]")
    private val bosId = tokenToId.getValue("[BOS]")
    private val eosId = tokenToId.getValue("[EOS]")
    private val unkId = tokenToId.getValue("[UNK]")

    fun encode(text: String, maxSeqLen: Int): IntArray {
        val ids = mutableListOf(bosId)
        for (char in text.lowercase()) {
            if (ids.size >= maxSeqLen - 1) break
            ids += tokenToId[char.toString()] ?: unkId
        }
        if (ids.size < maxSeqLen) {
            ids += eosId
        }
        return ids.toIntArray()
    }

    fun decode(tokenIds: IntArray): String =
        tokenIds
            .filterNot { it == bosId || it == eosId || it == padId }
            .joinToString(separator = "") { tokenForId(it) }

    fun tokenForId(id: Int): String = vocabulary.getOrElse(id) { "[UNK]" }

    fun displayLabel(id: Int): String = when (val token = tokenForId(id)) {
        " " -> "␠"
        "\n" -> "\\n"
        else -> token
    }

    fun displayLabels(tokenIds: IntArray): List<String> = tokenIds.map(::displayLabel)

    companion object {
        fun fromAsset(context: Context, assetName: String = "vocab.json"): CharTokenizer {
            val json = context.assets.open(assetName).bufferedReader().use { it.readText() }
            val array = JSONArray(json)
            val vocab = List(array.length()) { index -> array.getString(index) }
            return CharTokenizer(vocab)
        }
    }
}

class MiniTransformer private constructor(
    val config: ModelConfig,
    private val weights: TransformerWeights,
) {
    fun forward(
        tokenIds: IntArray,
        onTensor: (TensorSnapshot) -> Unit = {},
    ): ForwardResult {
        require(tokenIds.isNotEmpty()) { "forward() requires at least one token." }
        require(tokenIds.size <= config.maxSeqLen) {
            "Sequence length ${tokenIds.size} exceeds max_seq_len ${config.maxSeqLen}."
        }

        val tokenEmbeddings = Array(tokenIds.size) { position ->
            val tokenId = tokenIds[position]
            require(tokenId in 0 until config.vocabSize) { "Token id $tokenId is outside the vocabulary." }
            weights.embedding[tokenId].copyOf()
        }
        val positionEncoding = TensorUtils.sinusoidalPositionEncoding(tokenIds.size, config.dModel)
        var hidden = TensorUtils.add(tokenEmbeddings, positionEncoding)

        onTensor(TensorSnapshot.MatrixSnapshot("token_embeddings", TensorUtils.copy(tokenEmbeddings)))
        onTensor(TensorSnapshot.MatrixSnapshot("position_encoding", TensorUtils.copy(positionEncoding)))
        onTensor(TensorSnapshot.MatrixSnapshot("embedding_plus_position", TensorUtils.copy(hidden)))

        weights.layers.forEachIndexed { layerIndex, layer ->
            val query = TensorUtils.linear(hidden, layer.wQ)
            val key = TensorUtils.linear(hidden, layer.wK)
            val value = TensorUtils.linear(hidden, layer.wV)
            val scores = TensorUtils.scaledDotProductScores(query, key)
            val attentionWeights = TensorUtils.causalSoftmax(scores)
            val context = TensorUtils.matMul(attentionWeights, value)
            val projectedContext = TensorUtils.linear(context, layer.wO)
            val attentionResidual = TensorUtils.add(hidden, projectedContext)
            val afterAttention = TensorUtils.layerNorm(
                input = attentionResidual,
                gamma = layer.norm1Gamma,
                beta = layer.norm1Beta,
            )

            val ffHidden = TensorUtils.linear(afterAttention, layer.w1)
            val ffActivated = TensorUtils.map(ffHidden, TensorUtils::gelu)
            val ffProjected = TensorUtils.linear(ffActivated, layer.w2)
            val ffResidual = TensorUtils.add(afterAttention, ffProjected)
            hidden = TensorUtils.layerNorm(
                input = ffResidual,
                gamma = layer.norm2Gamma,
                beta = layer.norm2Beta,
            )

            onTensor(TensorSnapshot.MatrixSnapshot("layer_${layerIndex}_attention_scores", TensorUtils.copy(scores)))
            onTensor(TensorSnapshot.MatrixSnapshot("layer_${layerIndex}_attention_weights", TensorUtils.copy(attentionWeights)))
            onTensor(TensorSnapshot.MatrixSnapshot("layer_${layerIndex}_hidden_states", TensorUtils.copy(hidden)))
        }

        val logits = TensorUtils.tiedProjection(hidden.last(), weights.embedding)
        onTensor(TensorSnapshot.VectorSnapshot("final_logits", logits.copyOf()))

        return ForwardResult(
            tokenIds = tokenIds.copyOf(),
            finalHiddenStates = TensorUtils.copy(hidden),
            logits = logits,
        )
    }

    companion object {
        fun loadBundle(
            context: Context,
            weightsAssetName: String = "mini_transformer_weights.json",
            vocabAssetName: String = "vocab.json",
        ): ModelBundle {
            val tokenizer = CharTokenizer.fromAsset(context, vocabAssetName)
            val weightsJson = context.assets.open(weightsAssetName).bufferedReader().use { it.readText() }
            val weights = TransformerWeights.fromJson(weightsJson)
            require(tokenizer.size == weights.config.vocabSize) {
                "Tokenizer vocab size ${tokenizer.size} does not match weight file vocab size ${weights.config.vocabSize}."
            }
            return ModelBundle(
                tokenizer = tokenizer,
                model = MiniTransformer(weights.config, weights),
            )
        }
    }
}
