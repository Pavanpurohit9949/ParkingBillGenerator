from io import BytesIO

import qrcode


def build_upi_url(upi_id: str, merchant_name: str, amount: float, note: str) -> str:
    return (
        f"upi://pay?pa={upi_id}&pn={merchant_name}"
        f"&am={amount:.2f}&cu=INR&tn={note}"
    )


def generate_qr_png_bytes(payload: str) -> bytes:
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
