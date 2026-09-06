    live_story.append(live_summary)

    live_story.append(Paragraph("Multi-Agent Assessment", live_section))
    agent_rows = [["Agent", "State / Finding", "Detail"]]
    agent_rows.extend([
        ["Anomaly Agent", live_agents["anomaly"]["status"], f'{live_agents["anomaly"]["detections"]} detections; primary {live_agents["anomaly"]["primary_sensor"]}'],
        ["Sensor Agent", live_agents["sensor"]["top_sensor"], f'{live_agents["sensor"]["top_risk"]:.1f}/100 top sensor risk'],
        ["Trend Agent", live_agents["trend"]["direction"], f'{live_agents["trend"]["recent_risk"]:.1f} recent risk; slope {live_agents["trend"]["slope"]}'],
        ["Telemetry Health", live_agents["telemetry"]["status"], f'{live_agents["telemetry"]["quality"]}% quality'],
        ["Mission Agent", live_agents["mission"]["mission_state"], live_agents["mission"]["recommendation"]],
        ["Incident Agent", "ACTIVE" if live_agents["incident"]["active"] else "CLEAR", f'{live_agents["incident"]["event_count"]} incident event(s)'],
    ])
    agent_table = Table(agent_rows, colWidths=[40 * mm, 40 * mm, 88 * mm], repeatRows=1)
    agent_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F8FA")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    live_story.append(agent_table)
    live_story.append(Spacer(1, 7))

    live_story.append(Paragraph("Peak Risk Event", live_section))
    peak_time = float(peak["time_s"])
    peak_risk_value = float(peak["overall_risk"])
    peak_sensor = str(peak["primary_risk_sensor"])
    peak_text = (
        "Peak system risk was "
        f"<b>{peak_risk_value:.1f}/100</b> "
        f"at <b>{peak_time:.2f} s</b>. "
        f"<b>{peak_sensor}</b> was the primary risk sensor."
    )
    live_story.append(Paragraph(peak_text, live_body))

    incident_events = live_agents["incident"].get("events", [])
    if incident_events:
        live_story.append(Paragraph("Recent Critical Incidents", live_section))
        incident_rows = [["Start (s)", "End (s)", "Peak Risk", "Sensor"]]
        for event in incident_events:
            incident_rows.append([str(event["start_s"]), str(event["end_s"]), f'{event["peak_risk"]:.1f}/100', str(event["sensor"])])
        incident_table = Table(incident_rows, colWidths=[36 * mm, 36 * mm, 44 * mm, 52 * mm], repeatRows=1)
        incident_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEF3F2")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        live_story.append(incident_table)

    live_story.append(Spacer(1, 7))
    live_story.append(Paragraph("Prototype limitation: risk and agent outputs are analytical indicators and are not flight-safety guarantees.", live_small))
    live_doc.build(live_story)
    st.download_button(
        label="Download Live PDF Report",
        data=live_pdf.getvalue(),
        file_name="rocket_guardian_live_mission_report.pdf",
        mime="application/pdf",
    )