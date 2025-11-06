# Small helpers I reuse across scripts.

def parse_payload(s: str):
    # I accept strings like "00 7A 3F 01 00 00 00 00" or "007A3F0100000000".
    # I return exactly 8 integers (0..255).
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().replace(" ", "").replace("0x", "").replace("0X", "")
    if not s:
        return [0] * 8
    bytes_hex = [s[i:i+2] for i in range(0, len(s), 2)]
    while len(bytes_hex) < 8:
        bytes_hex.append("00")
    bytes_hex = bytes_hex[:8]
    out = []
    for b in bytes_hex:
        try:
            out.append(int(b, 16))
        except Exception:
            out.append(0)
    return out
