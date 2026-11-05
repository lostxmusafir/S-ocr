import urllib.request
import urllib.error
import json
import uuid
import os

def test_api_call(url, image_path):
    print(f"Testing URL: {url} ...")
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    body = bytearray()
    # file field
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(image_path)}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    # include_images field
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="include_images"\r\n\r\n')
    body.extend(b"true\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=35) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[SUCCESS {resp.status}]")
            print("  Final Code   :", data.get("final_code"))
            print("  State        :", data.get("parsed_info", {}).get("state_name"))
            print("  Model        :", data.get("parsed_info", {}).get("model_code"))
            print("  Year         :", data.get("parsed_info", {}).get("year"))
            print("  Serial No    :", data.get("parsed_info", {}).get("serial_no"))
            print("  Suffix       :", data.get("parsed_info", {}).get("suffix"))
            print("  Elapsed Time :", data.get("elapsed_ms"), "ms")
            print("  Digital Stamp:", "Included" if data.get("digital_stamp_base64") else "None")
    except urllib.error.HTTPError as e:
        print(f"[HTTP ERROR {e.code}]")
        print("  Body:", e.read().decode("utf-8"))
    except Exception as e:
        print("[REQUEST ERROR]:", str(e))
    print("=" * 60)

if __name__ == "__main__":
    test_img = "WhatsApp Image 2026-10-22 at 4.55.31 PM.jpeg"
    print("Simulating EXACT Postman multipart/form-data Request:\n")
    
    # 1. Render URL
    test_api_call("https://s-ocr-3zbt.onrender.com/ocr", test_img)
    
    # 2. Vercel URL
    test_api_call("https://s-ocr.vercel.app/api/ocr", test_img)
