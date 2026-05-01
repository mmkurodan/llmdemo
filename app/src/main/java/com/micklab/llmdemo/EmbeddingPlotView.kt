package com.micklab.llmdemo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

class EmbeddingPlotView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private var points: List<Point2D> = emptyList()
    private var labels: List<String> = emptyList()

    private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.LTGRAY
        strokeWidth = dp(1f)
    }
    private val pathPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(96, 63, 81, 181)
        strokeWidth = dp(2f)
        style = Paint.Style.STROKE
    }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.DKGRAY
        textSize = sp(12f)
    }
    private val placeholderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GRAY
        textSize = sp(14f)
        textAlign = Paint.Align.CENTER
    }

    fun submitEmbeddings(embeddings: Matrix, tokenLabels: List<String>) {
        points = TensorUtils.pca2D(embeddings)
        labels = tokenLabels
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.WHITE)

        if (points.isEmpty()) {
            canvas.drawText(
                "PCA projection of token embeddings will appear here",
                width / 2f,
                height / 2f,
                placeholderPaint,
            )
            return
        }

        val padding = dp(28f)
        val plotLeft = padding
        val plotTop = padding
        val plotRight = width - padding
        val plotBottom = height - padding

        var minX = points.minOf { it.x }
        var maxX = points.maxOf { it.x }
        var minY = points.minOf { it.y }
        var maxY = points.maxOf { it.y }

        if (maxX - minX < 1e-4f) {
            minX -= 1f
            maxX += 1f
        }
        if (maxY - minY < 1e-4f) {
            minY -= 1f
            maxY += 1f
        }

        val mapX: (Float) -> Float = { value ->
            plotLeft + (value - minX) / (maxX - minX) * (plotRight - plotLeft)
        }
        val mapY: (Float) -> Float = { value ->
            plotBottom - (value - minY) / (maxY - minY) * (plotBottom - plotTop)
        }

        val zeroX = mapX(0f).coerceIn(plotLeft, plotRight)
        val zeroY = mapY(0f).coerceIn(plotTop, plotBottom)
        canvas.drawLine(plotLeft, zeroY, plotRight, zeroY, axisPaint)
        canvas.drawLine(zeroX, plotTop, zeroX, plotBottom, axisPaint)

        for (index in 1 until points.size) {
            canvas.drawLine(
                mapX(points[index - 1].x),
                mapY(points[index - 1].y),
                mapX(points[index].x),
                mapY(points[index].y),
                pathPaint,
            )
        }

        for (index in points.indices) {
            val point = points[index]
            val x = mapX(point.x)
            val y = mapY(point.y)
            val colorAmount = index / max(1f, (points.size - 1).toFloat())
            pointPaint.color = Color.rgb(
                lerp(33, 244, colorAmount),
                lerp(150, 67, colorAmount),
                lerp(243, 54, colorAmount),
            )
            canvas.drawCircle(x, y, dp(5f), pointPaint)
            canvas.drawText(labels.getOrElse(index) { index.toString() }, x + dp(6f), y - dp(6f), labelPaint)
        }
    }

    private fun lerp(start: Int, end: Int, amount: Float): Int =
        (start + (end - start) * amount).toInt().coerceIn(0, 255)

    private fun dp(value: Float): Float = value * resources.displayMetrics.density

    private fun sp(value: Float): Float = value * resources.displayMetrics.scaledDensity
}
