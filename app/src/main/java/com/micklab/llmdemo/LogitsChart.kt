package com.micklab.llmdemo

import android.content.Context
import android.graphics.Color
import android.util.AttributeSet
import com.github.mikephil.charting.charts.BarChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import kotlin.math.max
import kotlin.math.min

data class LogitBar(val label: String, val value: Float)

class LogitsChart @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : BarChart(context, attrs) {
    init {
        description.isEnabled = false
        legend.isEnabled = false
        axisRight.isEnabled = false
        axisLeft.textColor = Color.DKGRAY
        axisLeft.setDrawZeroLine(true)
        axisLeft.zeroLineColor = Color.LTGRAY
        axisLeft.granularity = 0.5f
        xAxis.position = XAxis.XAxisPosition.BOTTOM
        xAxis.textColor = Color.DKGRAY
        xAxis.granularity = 1f
        xAxis.labelRotationAngle = -20f
        xAxis.setDrawGridLines(false)
        setNoDataText("Run the model to inspect next-token logits.")
        setNoDataTextColor(Color.GRAY)
        setTouchEnabled(false)
        setScaleEnabled(false)
        setPinchZoom(false)
        setFitBars(true)
    }

    fun submit(entries: List<LogitBar>) {
        if (entries.isEmpty()) {
            clear()
            invalidate()
            return
        }

        val barEntries = entries.mapIndexed { index, item ->
            BarEntry(index.toFloat(), item.value)
        }

        val dataSet = BarDataSet(barEntries, "Logits").apply {
            color = Color.rgb(63, 81, 181)
            valueTextColor = Color.DKGRAY
            valueTextSize = 10f
        }

        data = BarData(dataSet).apply {
            barWidth = 0.72f
        }
        xAxis.valueFormatter = IndexAxisValueFormatter(entries.map { it.label })
        xAxis.labelCount = entries.size

        val minValue = entries.minOf { it.value }
        val maxValue = entries.maxOf { it.value }
        axisLeft.axisMinimum = min(0f, minValue * 1.15f)
        axisLeft.axisMaximum = max(0.5f, maxValue * 1.15f)

        notifyDataSetChanged()
        invalidate()
    }
}
