import hashlib
from datetime import datetime

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from billing import calculate_amount
from database import (
    close_session,
    create_entry,
    get_open_session,
    init_db,
    list_active_sessions,
    mark_paid,
)
from ocr import extract_vehicle_number, normalize_plate
from qr_payment import build_upi_url, generate_qr_png_bytes
from video_gate import detect_from_offline_video

UPI_ID = "yourparking@upi"
MERCHANT_NAME = "Secunderabad Parking"


def minutes_between(start_iso: str, end_dt: datetime) -> int:
    start_dt = datetime.fromisoformat(start_iso)
    return max(1, int((end_dt - start_dt).total_seconds() // 60))


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def run_video_detection(tmp_path: str):
    """Exit / quick path: minimal frames (local CPU)."""
    return detect_from_offline_video(tmp_path, instant=True)


def run_entry_video_scan(tmp_path: str):
    """Entry path: YOLO + multi-frame plate voting (local CPU)."""
    return detect_from_offline_video(
        tmp_path,
        max_frames=40,
        frame_stride=8,
        resize_width=640,
        instant=False,
    )


def render_entry():
    st.subheader("Vehicle Entry")
    mode = st.radio(
        "Entry Input Mode",
        ["Manual (image/OCR)", "Upload entry video (auto)"],
        horizontal=True,
    )

    if mode == "Manual (image/OCR)":
        vehicle_type = st.selectbox("Vehicle Type", ["2W", "4W", "HEAVY"], index=1, key="entry_vehicle_manual")
        image_file = st.file_uploader(
            "Upload number plate image (optional)",
            type=["jpg", "jpeg", "png"],
            key="entry_upload_manual",
        )

        detected_plate = ""
        if image_file:
            detected_plate = extract_vehicle_number(image_file.read())
            if detected_plate:
                st.success(f"Detected Number: {detected_plate}")
            else:
                st.warning("Could not read plate clearly. Enter manually.")

        plate_input = st.text_input(
            "Vehicle Number",
            value=detected_plate,
            placeholder="e.g. TS09AB1234",
            key="entry_plate_manual",
        )

        if st.button("Record Entry", type="primary"):
            plate = normalize_plate(plate_input)
            if not plate:
                st.error("Vehicle number is required.")
                return
            if get_open_session(plate):
                st.error("This vehicle already has an active parking session.")
                return

            create_entry(plate, vehicle_type, datetime.now())
            st.success(f"Entry recorded for {plate}.")
        return

    # Auto video mode — scan + record entry automatically when a new file is uploaded
    video_file = st.file_uploader(
        "Upload entry video (scan & record run automatically)",
        type=["mp4", "avi", "mov", "mkv"],
        key="entry_video_upload",
    )
    st.caption(
        "**YOLO** vehicle type + **multi-frame** plate voting (local CPU). "
        "If a plate is read with sufficient confidence and no duplicate session exists, **entry is saved automatically**."
    )

    if "entry_vehicle_type_auto" not in st.session_state:
        st.session_state["entry_vehicle_type_auto"] = "4W"
    if "entry_plate_auto" not in st.session_state:
        st.session_state["entry_plate_auto"] = ""

    if not video_file:
        st.session_state.pop("entry_video_process_key", None)
    else:
        buf = video_file.getbuffer()
        sig = hashlib.sha256(bytes(buf)).digest().hex()
        process_key = sig
        if process_key != st.session_state.get("entry_video_process_key"):
            with st.spinner("Scanning video (precise)..."):
                tmp_path = save_uploaded_file(video_file)
                try:
                    result = run_entry_video_scan(tmp_path)
                except (RuntimeError, ValueError) as e:
                    st.error(str(e))
                else:
                    vt = result.vehicle_type if result.vehicle_type in ("2W", "4W", "HEAVY") else "4W"
                    st.session_state["entry_vehicle_type_auto"] = vt
                    st.session_state["entry_plate_auto"] = result.vehicle_number or ""
                    st.session_state["entry_video_process_key"] = process_key

                    plate = normalize_plate(result.vehicle_number or "")
                    min_conf = 48.0
                    conf_ok = result.plate_confidence >= min_conf
                    if plate and conf_ok and not get_open_session(plate):
                        create_entry(plate, vt, datetime.now())
                        st.success(
                            f"**Entry saved automatically** for `{plate}` ({vt}). "
                            f"Plate confidence `{result.plate_confidence:.0f}` / vehicle `{result.vehicle_confidence:.0f}`."
                        )
                    elif plate and conf_ok and get_open_session(plate):
                        st.warning(
                            f"Detected `{plate}` but this vehicle **already has an active session**. "
                            "No new entry created."
                        )
                        st.info(f"Detected type `{vt}` (conf `{result.vehicle_confidence:.0f}`).")
                    elif plate and not conf_ok:
                        st.info(
                            f"Detected `{plate}` with lower confidence (`{result.plate_confidence:.0f}`). "
                            "Review below and tap **Record entry** to save."
                        )
                        st.write(f"Vehicle type guess: `{vt}` (conf `{result.vehicle_confidence:.0f}`).")
                    else:
                        st.warning("Could not read a plate from this video. Enter it manually below, then **Record entry**.")

    _vt_opts = ["2W", "4W", "HEAVY"]
    _vt_cur = st.session_state["entry_vehicle_type_auto"]
    _vt_i = _vt_opts.index(_vt_cur) if _vt_cur in _vt_opts else 1
    vehicle_type = st.selectbox(
        "Vehicle Type (auto-filled from video)",
        _vt_opts,
        index=_vt_i,
        key="entry_vehicle_type_auto",
    )
    plate_input = st.text_input(
        "Vehicle Number (auto-filled from video)",
        value=st.session_state["entry_plate_auto"],
        placeholder="e.g. TS09AB1234",
        key="entry_plate_auto",
    )

    if st.button("Record Entry", type="primary", key="entry_record_auto"):
        plate = normalize_plate(plate_input)
        if not plate:
            st.error("Vehicle number is required.")
            return
        if get_open_session(plate):
            st.error("This vehicle already has an active parking session.")
            return

        create_entry(plate, vehicle_type, datetime.now())
        st.success(f"Entry recorded for {plate}.")


def render_exit():
    st.subheader("Vehicle Exit")
    mode = st.radio(
        "Exit Input Mode",
        ["Manual (image/OCR)", "Upload exit video (auto)"],
        horizontal=True,
    )

    if mode == "Manual (image/OCR)":
        image_file = st.file_uploader(
            "Upload number plate image (optional)",
            type=["jpg", "jpeg", "png"],
            key="exit_upload_manual",
        )

        detected_plate = ""
        if image_file:
            detected_plate = extract_vehicle_number(image_file.read())
            if detected_plate:
                st.success(f"Detected Number: {detected_plate}")
            else:
                st.warning("Could not read plate clearly. Enter manually.")

        plate_input = st.text_input(
            "Vehicle Number",
            value=detected_plate,
            placeholder="e.g. TS09AB1234",
            key="exit_plate_manual",
        )

        if st.button("Generate Bill", key="exit_generate_manual"):
            plate = normalize_plate(plate_input)
            if not plate:
                st.error("Vehicle number is required.")
                return

            session = get_open_session(plate)
            if not session:
                st.error("No active session found for this vehicle.")
                return

            exit_dt = datetime.now()
            duration_minutes = minutes_between(session["entry_time"], exit_dt)
            amount = calculate_amount(session["vehicle_type"], duration_minutes)
            close_session(session["id"], exit_dt, duration_minutes, amount)

            st.success("Bill generated successfully.")
            st.write(f"Vehicle: `{session['vehicle_number']}`")
            st.write(f"Duration: `{duration_minutes} minutes`")
            st.write(f"Amount: `Rs {amount:.2f}`")

            upi_note = f"Parking-{session['id']}-{session['vehicle_number']}"
            upi_url = build_upi_url(UPI_ID, MERCHANT_NAME, amount, upi_note)
            qr_bytes = generate_qr_png_bytes(upi_url)
            st.image(qr_bytes, caption="Scan to Pay (UPI)", width=240)

            if st.button("Mark Payment Received", key="exit_mark_paid_manual"):
                mark_paid(
                    session["id"],
                    payment_ref=f"UPI-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                )
                st.success("Payment marked as PAID.")
        return

    # Auto video mode
    video_file = st.file_uploader(
        "Upload exit video",
        type=["mp4", "avi", "mov", "mkv"],
        key="exit_video_upload",
    )
    st.caption(
        "Video is scanned **automatically** (fast): up to **3** frames, OCR only (local CPU)."
    )

    if "exit_plate_auto" not in st.session_state:
        st.session_state["exit_plate_auto"] = ""

    if st.button("Auto-detect from video", key="exit_autodetect"):
        if not video_file:
            st.error("Please upload an exit video first.")
            return

        with st.spinner("Scanning video..."):
            tmp_path = save_uploaded_file(video_file)
            try:
                result = run_video_detection(tmp_path)
            except (RuntimeError, ValueError) as e:
                st.error(str(e))
                return
        st.session_state["exit_plate_auto"] = result.vehicle_number

        st.success("Auto-detection finished.")
        if result.vehicle_number:
            st.write(f"Detected plate: `{result.vehicle_number}` (conf: `{result.plate_confidence:.1f}`)")
        else:
            st.warning("Could not confidently read plate from video. You can edit manually.")

    plate_input = st.text_input(
        "Vehicle Number (auto-filled from video)",
        value=st.session_state["exit_plate_auto"],
        placeholder="e.g. TS09AB1234",
        key="exit_plate_auto",
    )

    if st.button("Generate Bill", key="exit_generate_auto"):
        plate = normalize_plate(plate_input)
        if not plate:
            st.error("Vehicle number is required.")
            return

        session = get_open_session(plate)
        if not session:
            st.error("No active session found for this vehicle.")
            return

        exit_dt = datetime.now()
        duration_minutes = minutes_between(session["entry_time"], exit_dt)
        amount = calculate_amount(session["vehicle_type"], duration_minutes)
        close_session(session["id"], exit_dt, duration_minutes, amount)

        st.success("Bill generated successfully.")
        st.write(f"Vehicle: `{session['vehicle_number']}`")
        st.write(f"Duration: `{duration_minutes} minutes`")
        st.write(f"Amount: `Rs {amount:.2f}`")

        upi_note = f"Parking-{session['id']}-{session['vehicle_number']}"
        upi_url = build_upi_url(UPI_ID, MERCHANT_NAME, amount, upi_note)
        qr_bytes = generate_qr_png_bytes(upi_url)
        st.image(qr_bytes, caption="Scan to Pay (UPI)", width=240)

        if st.button("Mark Payment Received", key="exit_mark_paid_auto"):
            mark_paid(
                session["id"],
                payment_ref=f"UPI-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            )
            st.success("Payment marked as PAID.")


def render_dashboard():
    st.subheader("Active Parked Vehicles")
    rows = list_active_sessions()
    if not rows:
        st.info("No active parked vehicles currently.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Parking Bill Generator", layout="wide")
    init_db()

    st.title("Automated Parking Bill Generator")
    st.caption("Entry/Exit, billing, and QR payment for large parking lots.")

    tab1, tab2, tab3 = st.tabs(["Entry", "Exit & Billing", "Dashboard"])
    with tab1:
        render_entry()
    with tab2:
        render_exit()
    with tab3:
        render_dashboard()


if __name__ == "__main__":
    main()
