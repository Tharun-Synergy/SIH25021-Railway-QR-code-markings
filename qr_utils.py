import uuid
import qrcode
import io

def generate_unique_string():
    return str(uuid.uuid4())

def generate_qr_code(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

def generate_gcode(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.modules
    gcode = ["G21", "G90", "G00 Z5"]
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell:
                gcode += [f"G00 X{x} Y{y}", "G01 Z-1 F100", "G00 Z5"]
    gcode.append("M30")
    buffer = io.BytesIO()
    buffer.write('\n'.join(gcode).encode())
    buffer.seek(0)
    return buffer