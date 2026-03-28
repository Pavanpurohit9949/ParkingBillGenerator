# Parking Bill Generator (MVP)

Automated parking management software for large lots (for example railway station parking) with:

- Vehicle number plate scan (OCR from image)
- Entry and exit timestamp recording
- Auto parking duration and bill calculation
- QR code payment using UPI link
- Active vehicles dashboard

## Features

1. **Vehicle Entry**
   - Upload number plate image (optional) and auto-detect vehicle number.
   - Manual correction is supported.
   - Stores entry time in local SQLite database.

2. **Vehicle Exit and Billing**
   - Detect or enter vehicle number.
   - Calculates parking duration and amount based on vehicle type.
   - Generates UPI payment QR instantly.
   - Payment can be marked as paid.

3. **Dashboard**
   - Shows all currently parked vehicles with entry times.

## Tech Stack

- Python
- Streamlit (UI)
- SQLite (data storage)
- OpenCV + Tesseract OCR (number plate text extraction)
- qrcode (payment QR generation)

## Setup

1. Open terminal in this project folder.
2. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Install **Tesseract OCR** engine on Windows and add it to PATH:
   - https://github.com/UB-Mannheim/tesseract/wiki

5. Run the app:

```powershell
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (fastest)

This repo includes **`packages.txt`** so Linux workers install **Tesseract** and libraries **OpenCV** needs. **`requirements.txt`** is used for Python packages.

### Steps

1. Put the project on **GitHub** (Streamlit Community Cloud pulls from Git).
2. Sign in at [Streamlit Community Cloud](https://streamlit.io/cloud) and click **Create app**.
3. Pick the repo, branch, and set **Main file** to `app.py`.
4. Deploy. The first run may be slow while **YOLO** downloads weights.

### SQLite on Cloud: important limitation

On Community Cloud the app filesystem is **not durable** for long-term storage. Your **`parking.db`** (default) can be **reset** when the app restarts, sleeps, or redeploys. That is fine for a **demo** or quick test.

For **real parking data** that must persist, use a **hosted database** (for example **Neon**, **Supabase**, or **Railway PostgreSQL**) and point the app at it. This MVP still uses **SQLite** only; switching to Postgres requires a small code change to use that database instead of `parking.db`.

### Docker (persistent SQLite on a volume)

For a single container host where you can mount a disk, see the **`Dockerfile`** and set **`PARKING_DB_PATH`** to a file on the mounted volume (see earlier deploy notes in your workflow).

## Pricing Logic

- 2W: Rs 20/hour
- 4W: Rs 50/hour
- Heavy Vehicle: Rs 90/hour
- Billing is rounded up to full hours with minimum 1 hour.

Edit rates in `billing.py`.

## Production Notes

- Replace `UPI_ID` in `app.py` with your real UPI merchant ID.
- For real deployment, add:
  - Camera integration at gate
  - Automatic exit barrier control
  - Online payment webhook verification
  - Multi-user authentication and audit logs
  - Cloud/PostgreSQL backend and backup
