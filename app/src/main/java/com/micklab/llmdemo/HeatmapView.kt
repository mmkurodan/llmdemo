package com.micklab.llmdemo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class HeatmapView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private var matrix: Matrix = emptyArray()
    private var labels: List<String> = emptyList()

    private val cellPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(48, 33, 33, 33)
        style = Paint.Style.STROKE
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.DKGRAY
        textSize = sp(12f)
    }
    private val placeholderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GRAY
        textAlign = Paint.Align.CENTER
        textSize = sp(14f)
    }

    fun submit(attention: Matrix, tokenLabels: List<String>) {
        matrix = TensorUtils.copy(attention)
        labels = tokenLabels
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.WHITE)

        if (matrix.isEmpty()) {
            canvas.drawText(
                "Attention heatmap will appear here",
                width / 2f,
                height / 2f,
                placeholderPaint,
            )
            return
        }

        val rows = matrix.size
        val cols = matrix[0].size
        val leftPad = dp(64f)
        val topPad = dp(48f)
        val rightPad = dp(16f)
        val bottomPad = dp(24f)
        val contentWidth = (width - leftPad - rightPad).coerceAtLeast(1f)
        val contentHeight = (height - topPad - bottomPad).coerceAtLeast(1f)
        val cellWidth = contentWidth / cols
        val cellHeight = contentHeight / rows

        var minValue = Float.POSITIVE_INFINITY
        var maxValue = Float.NEGATIVE_INFINITY
        for (row in matrix.indices) {
            for (col in matrix[row].indices) {
                minValue = min(minValue, matrix[row][col])
                maxValue = max(maxValue, matrix[row][col])
            }
        }
        val range = (maxValue - minValue).coerceAtLeast(1e-6f)

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val normalized = ((matrix[row][col] - minValue) / range).coerceIn(0f, 1f)
                cellPaint.color = heatColor(normalized)

                val left = leftPad + col * cellWidth
                val top = topPad + row * cellHeight
                val right = left + cellWidth
                val bottom = top + cellHeight

                canvas.drawRect(left, top, right, bottom, cellPaint)
                canvas.drawRect(left, top, right, bottom, borderPaint)
            }
        }

        labelPaint.textAlign = Paint.Align.RIGHT
        for (row in 0 until rows.coerceAtMost(labels.size)) {
            val y = topPad + row * cellHeight + cellHeight / 2f + labelPaint.textSize / 3f
            canvas.drawText(labels[row], leftPad - dp(8f), y, labelPaint)
        }

        labelPaint.textAlign = Paint.Align.CENTER
        for (col in 0 until cols.coerceAtMost(labels.size)) {
            val x = leftPad + col * cellWidth + cellWidth / 2f
            canvas.drawText(labels[col], x, topPad - dp(10f), labelPaint)
        }
    }

    private fun heatColor(value: Float): Int {
        val r = lerp(230, 244, value)
        val g = lerp(240, 67, value)
        val b = lerp(255, 54, value)
        return Color.rgb(r, g, b)
    }

    private fun lerp(start: Int, end: Int, amount: Float): Int =
        (start + (end - start) * amount).roundToInt().coerceIn(0, 255)

    private fun dp(value: Float): Float = value * resources.displayMetrics.density

    private fun sp(value: Float): Float = value * resources.displayMetrics.scaledDensity
}
