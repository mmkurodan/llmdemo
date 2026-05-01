package com.micklab.llmdemo

import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

typealias Matrix = Array<FloatArray>

data class Point2D(val x: Float, val y: Float)

object TensorUtils {
    fun zeros(rows: Int, cols: Int): Matrix = Array(rows) { FloatArray(cols) }

    fun copy(matrix: Matrix): Matrix = Array(matrix.size) { matrix[it].copyOf() }

    fun add(left: Matrix, right: Matrix): Matrix {
        require(left.size == right.size) { "Matrix row counts do not match." }
        if (left.isEmpty()) return emptyArray()
        require(left[0].size == right[0].size) { "Matrix column counts do not match." }

        return Array(left.size) { row ->
            FloatArray(left[row].size) { col ->
                left[row][col] + right[row][col]
            }
        }
    }

    fun linear(input: Matrix, weight: Matrix): Matrix {
        if (input.isEmpty()) return emptyArray()
        val inputDim = input[0].size
        require(weight.size == inputDim) {
            "Linear weight expects input dim ${weight.size}, but received $inputDim."
        }

        val outputDim = if (weight.isEmpty()) 0 else weight[0].size
        val result = zeros(input.size, outputDim)

        for (row in input.indices) {
            for (outIndex in 0 until outputDim) {
                var sum = 0f
                for (inIndex in 0 until inputDim) {
                    sum += input[row][inIndex] * weight[inIndex][outIndex]
                }
                result[row][outIndex] = sum
            }
        }

        return result
    }

    fun scaledDotProductScores(query: Matrix, key: Matrix): Matrix {
        if (query.isEmpty() || key.isEmpty()) return emptyArray()
        val dimension = query[0].size
        require(key[0].size == dimension) { "Query and key dimensions must match." }

        val scale = sqrt(dimension.toFloat())
        val scores = zeros(query.size, key.size)

        for (row in query.indices) {
            for (col in key.indices) {
                var dot = 0f
                for (index in 0 until dimension) {
                    dot += query[row][index] * key[col][index]
                }
                scores[row][col] = dot / scale
            }
        }

        return scores
    }

    fun causalSoftmax(scores: Matrix): Matrix {
        if (scores.isEmpty()) return emptyArray()
        val result = zeros(scores.size, scores[0].size)

        for (row in scores.indices) {
            var maxScore = Float.NEGATIVE_INFINITY
            for (col in 0..row.coerceAtMost(scores[row].lastIndex)) {
                maxScore = max(maxScore, scores[row][col])
            }

            var sum = 0f
            for (col in scores[row].indices) {
                if (col > row) {
                    result[row][col] = 0f
                    continue
                }
                val value = kotlin.math.exp(scores[row][col] - maxScore)
                result[row][col] = value
                sum += value
            }

            if (sum > 0f) {
                for (col in 0..row.coerceAtMost(scores[row].lastIndex)) {
                    result[row][col] /= sum
                }
            }
        }

        return result
    }

    fun matMul(left: Matrix, right: Matrix): Matrix {
        if (left.isEmpty() || right.isEmpty()) return emptyArray()
        require(left[0].size == right.size) { "Inner matrix dimensions do not align." }

        val result = zeros(left.size, right[0].size)
        for (row in left.indices) {
            for (col in right[0].indices) {
                var sum = 0f
                for (index in right.indices) {
                    sum += left[row][index] * right[index][col]
                }
                result[row][col] = sum
            }
        }
        return result
    }

    fun map(matrix: Matrix, transform: (Float) -> Float): Matrix =
        Array(matrix.size) { row ->
            FloatArray(matrix[row].size) { col ->
                transform(matrix[row][col])
            }
        }

    fun gelu(value: Float): Float {
        val cubic = value * value * value
        val inner = 0.7978845608f * (value + 0.044715f * cubic)
        return 0.5f * value * (1f + kotlin.math.tanh(inner))
    }

    fun layerNorm(
        input: Matrix,
        gamma: FloatArray,
        beta: FloatArray,
        epsilon: Float = 1e-5f,
    ): Matrix {
        if (input.isEmpty()) return emptyArray()
        require(gamma.size == input[0].size && beta.size == input[0].size) {
            "Layer norm vectors must match hidden size."
        }

        return Array(input.size) { row ->
            val vector = input[row]
            val mean = vector.average().toFloat()
            var variance = 0f
            for (value in vector) {
                val delta = value - mean
                variance += delta * delta
            }
            variance /= vector.size.toFloat()

            val invStd = 1f / sqrt(variance + epsilon)
            FloatArray(vector.size) { index ->
                ((vector[index] - mean) * invStd * gamma[index]) + beta[index]
            }
        }
    }

    fun sinusoidalPositionEncoding(sequenceLength: Int, modelDim: Int): Matrix {
        return Array(sequenceLength) { position ->
            FloatArray(modelDim) { index ->
                val exponent = (2.0 * (index / 2)) / modelDim.toDouble()
                val denominator = 10000.0.pow(exponent)
                val angle = position / denominator
                if (index % 2 == 0) sin(angle).toFloat() else cos(angle).toFloat()
            }
        }
    }

    fun tiedProjection(vector: FloatArray, embeddingTable: Matrix): FloatArray {
        if (embeddingTable.isEmpty()) return FloatArray(0)
        require(vector.size == embeddingTable[0].size) {
            "Vector dimension ${vector.size} must match embedding dimension ${embeddingTable[0].size}."
        }

        return FloatArray(embeddingTable.size) { tokenId ->
            dot(vector, embeddingTable[tokenId])
        }
    }

    fun argmax(values: FloatArray): Int {
        var bestIndex = 0
        var bestValue = Float.NEGATIVE_INFINITY
        for (index in values.indices) {
            if (values[index] > bestValue) {
                bestValue = values[index]
                bestIndex = index
            }
        }
        return bestIndex
    }

    fun topK(values: FloatArray, k: Int): IntArray =
        values.indices
            .sortedByDescending { values[it] }
            .take(k.coerceAtMost(values.size))
            .toIntArray()

    fun pca2D(matrix: Matrix): List<Point2D> {
        if (matrix.isEmpty()) return emptyList()
        if (matrix.size == 1) return listOf(Point2D(0f, 0f))

        val cols = matrix[0].size
        val centered = center(matrix)
        if (cols == 1) {
            return centered.map { Point2D(it[0], 0f) }
        }

        val covariance = covariance(centered)
        val first = powerIteration(covariance)
        val deflated = deflate(covariance, first)
        val second = powerIteration(deflated)

        return centered.map { row ->
            Point2D(dot(row, first), dot(row, second))
        }
    }

    private fun center(matrix: Matrix): Matrix {
        val cols = matrix[0].size
        val means = FloatArray(cols)
        for (row in matrix.indices) {
            for (col in 0 until cols) {
                means[col] += matrix[row][col]
            }
        }
        for (col in 0 until cols) {
            means[col] /= matrix.size.toFloat()
        }

        return Array(matrix.size) { row ->
            FloatArray(cols) { col ->
                matrix[row][col] - means[col]
            }
        }
    }

    private fun covariance(centered: Matrix): Matrix {
        val cols = centered[0].size
        val covariance = zeros(cols, cols)
        val divisor = max(1, centered.size - 1).toFloat()

        for (row in centered.indices) {
            for (left in 0 until cols) {
                for (right in 0 until cols) {
                    covariance[left][right] += centered[row][left] * centered[row][right] / divisor
                }
            }
        }
        return covariance
    }

    private fun powerIteration(matrix: Matrix, iterations: Int = 32): FloatArray {
        if (matrix.isEmpty()) return FloatArray(0)

        var vector = FloatArray(matrix.size) { index ->
            if (index == 0) 1f else 1f / matrix.size.toFloat()
        }

        repeat(iterations) {
            val next = FloatArray(matrix.size)
            for (row in matrix.indices) {
                var sum = 0f
                for (col in matrix[row].indices) {
                    sum += matrix[row][col] * vector[col]
                }
                next[row] = sum
            }

            var norm = 0f
            for (value in next) {
                norm += value * value
            }
            norm = sqrt(norm).coerceAtLeast(1e-6f)

            for (index in next.indices) {
                next[index] /= norm
            }
            vector = next
        }

        return vector
    }

    private fun deflate(matrix: Matrix, eigenVector: FloatArray): Matrix {
        if (matrix.isEmpty()) return emptyArray()

        val transformed = FloatArray(matrix.size)
        for (row in matrix.indices) {
            var sum = 0f
            for (col in matrix[row].indices) {
                sum += matrix[row][col] * eigenVector[col]
            }
            transformed[row] = sum
        }

        val eigenValue = dot(eigenVector, transformed)
        return Array(matrix.size) { row ->
            FloatArray(matrix[row].size) { col ->
                matrix[row][col] - eigenValue * eigenVector[row] * eigenVector[col]
            }
        }
    }

    private fun dot(left: FloatArray, right: FloatArray): Float {
        require(left.size == right.size) { "Vector dimensions must match." }
        var sum = 0f
        for (index in left.indices) {
            sum += left[index] * right[index]
        }
        return sum
    }
}
