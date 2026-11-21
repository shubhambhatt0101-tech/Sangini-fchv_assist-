<!DOCTYPE html>
<html style="color-scheme:dark" lang="ne">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FCHV Registration — FCHV_assist</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#537abe;--bg1:hsla(202, 85%, 97%, 0.82);--bg2:#32ea42;--bg3:#1057e6;
  --line:lab(47.95% 73.56 60.84 / 0.08);--line2:rgba(13, 12, 12, 0.14);
  --text:#020303;--text2:#11112d;--muted:#4e525a;--muted2:#5D6A86;
  --red:#e41a1a;--red-t:rgba(232, 77, 77, 0.12);--red-t2:rgba(224, 36, 36, 0.2);
  --teal:#dd1010;--teal-t:rgba(89, 117, 243, 0.13);
  --r:10px;--rsm:7px;--rlg:16px;
  --sh:0 1px 3px rgba(0,0,0,0.45), 0 4px 14px rgba(0,0,0,0.35);
  --sh2:0 2px 8px rgba(132, 103, 103, 0.5), 0 8px 28px rgba(252, 118, 118, 0.4);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

body{font-family:'DM Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;}

.wrap{width:100%;max-width:440px;}

.brand{text-align:center;margin-bottom:24px;}

.logo-mark{width:44px;height:44px;border-radius:12px;background:var(--red);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:600;color:#150d0d;font-family:'DM Mono',monospace;margin:0 auto 10px;}
.brand-name{font-size:20px;font-weight:600;letter-spacing:-.3px;}
.brand-name span{color:var(--red);}
.brand-sub{font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;font-family:'DM Mono',monospace;margin-top:3px;}
.card{background:var(--bg1);border:1px solid var(--line2);border-radius:var(--rlg);padding:26px 26px 22px;box-shadow:var(--sh2);}
.card-title{font-size:15px;font-weight:600;color:var(--text);margin-bottom:4px;}
.card-sub{font-size:12px;color:var(--muted);margin-bottom:22px;font-family:'Noto Sans Devanagari',sans-serif;}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.field{margin-bottom:14px;}
.label{font-size:11px;font-weight:600;color:var(--text2);display:block;margin-bottom:5px;letter-spacing:.02em;font-family:'Noto Sans Devanagari',sans-serif;}
.req{color:var(--red);margin-left:2px;}
.input,.select{width:100%;padding:9px 12px;border-radius:var(--rsm);border:1.5px solid var(--line2);background:var(--bg1);font-size:13.5px;color:var(--text);outline:none;font-family:'DM Sans',sans-serif;transition:border-color .15s;}
.input:focus,.select:focus{border-color:rgba(99, 31, 31, 0.35);}
.input::placeholder{color:var(--muted2);}
.pw-wrap{position:relative;}
.pw-toggle{position:absolute;right:9px;top:50%;transform:translateY(-50%);background:none;border:none;cursor:pointer;color:var(--muted);font-size:12px;padding:3px;}
.id-preview{background:var(--teal-t);
    border:1px solid rgba(37, 30, 99, 0.18);
    border-radius:var(--rsm);
    padding:9px 12px;
    margin-bottom:14px;
    display:flex;
    align-items:center;
    gap:10px;}
    
.id-lbl{font-size:9px;color:var(--teal);font-weight:600;letter-spacing:.09em;font-family:'DM Mono',monospace;}

.id-val{font-size:16px;font-weight:700;color:var(--teal);font-family:'DM Mono',monospace;letter-spacing:-.3px;}

.btn{width:100%;padding:11px;border-radius:var(--r);border:none;font-size:14px;font-weight:600;cursor:pointer;transition:all .15s;font-family:'Noto Sans Devanagari',sans-serif;display:flex;align-items:center;justify-content:center;gap:7px;}
.btn-primary{background:var(--red);color:#fff;box-shadow:0 2px 9px rgba(224,82,82,0.22);}

.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 14px rgba(224,82,82,0.32);}

.btn-primary:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none;}

.spin{width:14px;height:14px;border-radius:50%;border:2px solid rgba(225, 178, 178, 0.35);border-top-color:#fff;animation:spin .7s linear infinite;display:none;}
@keyframes spin{to{transform:rotate(360deg)}}
.err-box{background:rgba(224,82,82,0.12);border:1px solid rgba(225, 48, 48, 0.22);border-radius:var(--rsm);padding:9px 12px;font-size:12px;color:#FF9A9A;margin-bottom:14px;display:none;font-family:'Noto Sans Devanagari',sans-serif;}
.err-box.on{display:block;animation:shake .3s ease;}
@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-4px)}75%{transform:translateX(4px)}}
.ok-box{background:var(--teal-t);border:1px solid rgba(34,168,141,.22);border-radius:var(--rsm);padding:9px 12px;font-size:12px;color:var(--teal);margin-bottom:14px;display:none;font-family:'Noto Sans Devanagari',sans-serif;}
.ok-box.on{display:block;}
.divider{display:flex;align-items:center;gap:10px;margin:16px 0;}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:var(--line2);}
.divider span{font-size:10px;color:var(--muted2);font-family:'DM Mono',monospace;}
.login-link{text-align:center;font-size:12px;color:var(--muted);}
.login-link a{color:var(--red);text-decoration:none;font-weight:600;}
.login-link a:hover{text-decoration:underline;}
.footer-note{text-align:center;font-size:10px;color:var(--muted2);margin-top:16px;font-family:'DM Mono',monospace;}
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">
    <div class="logo-mark">F</div>
    <div class="brand-name">FCHV<span>_assist</span></div>
    <div class="brand-sub">नयाँ FCHV खाता बनाउनुहोस्</div>
  </div>

  <div class="card">
    <div class="card-title">FCHV खाता दर्ता</div>
    <div class="card-sub">तपाईंको व्यक्तिगत FCHV खाता बनाउनुहोस् — आफ्ना बिरामीको रेकर्ड अलग राख्न।</div>

    <div class="err-box" id="errBox"></div>
    <div class="ok-box" id="okBox"></div>

    <div class="id-preview">
      <div>
        <div class="id-lbl">तपाईंको FCHV ID</div>
        <div class="id-val" id="idPreview">FCHV-…</div>
      </div>
    </div>

    <form id="regForm">
      <div class="row2">
        <div class="field">
          <label class="label">पूरा नाम <span class="req">*</span></label>
          <input class="input" type="text" id="fullName" placeholder="नाम थर" required>
        </div>
        <div class="field">
          <label class="label">फोन नम्बर <span class="req">*</span></label>
          <input class="input" type="tel" id="phone" placeholder="+977-98..." required>
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label class="label">जिल्ला <span class="req">*</span></label>
          <select class="select" id="district" required>
            <option value="">छान्नुहोस्</option>
            <option>Dhankuta</option><option>Sunsari</option><option>Morang</option>
            <option>Kathmandu</option><option>Lalitpur</option><option>Bhaktapur</option>
            <option>Kaski</option><option>Makwanpur</option><option>Chitwan</option>
            <option>Rupandehi</option><option>Palpa</option><option>Syangja</option>
            <option>Other</option>
          </select>
        </div>
        <div class="field">
          <label class="label">VDC / नगरपालिका</label>
          <input class="input" type="text" id="vdc" placeholder="VDC वा नगर">
        </div>
      </div>
      <div class="field">
        <label class="label">वडा नम्बर</label>
        <input class="input" type="text" id="ward" placeholder="वडा नं.">
      </div>
      <div class="row2">
        <div class="field">
          <label class="label">पासवर्ड <span class="req">*</span></label>
          <div class="pw-wrap">
            <input class="input" type="password" id="pwd" placeholder="कम्तिमा ८ अक्षर" required style="padding-right:34px">
            <button type="button" class="pw-toggle" onclick="togglePwd('pwd','pt1')">👁</button>
          </div>
        </div>
        <div class="field">
          <label class="label">पासवर्ड फेरि <span class="req">*</span></label>
          <div class="pw-wrap">
            <input class="input" type="password" id="pwd2" placeholder="दोहोर्याउनुहोस्" required style="padding-right:34px">
            <button type="button" class="pw-toggle" id="pt1" onclick="togglePwd('pwd2','pt1')">👁</button>
          </div>
        </div>
      </div>

      <button type="submit" class="btn btn-primary" id="regBtn">
        <span id="btnTxt">खाता बनाउनुहोस्</span>
        <div class="spin" id="btnSpin"></div>
      </button>
    </form>

    <div class="divider"><span>OR</span></div>
    <div class="login-link">पहिले नै खाता छ? <a href="/login">लगइन गर्नुहोस्</a></div>
  </div>
  <div class="footer-note">Harvard 2026 · FCHV_assist</div>
</div>

<script>
let counter = null;

async function fetchCounter() {
  try {
    const r = await fetch('/fchv/next_id');
    const d = await r.json();
    counter = d.next_number || 1;
    document.getElementById('idPreview').textContent = 'FCHV-' + String(counter).padStart(3, '0');
  } catch { document.getElementById('idPreview').textContent = 'FCHV-AUTO'; }
}

function togglePwd(inputId) {
  const inp = document.getElementById(inputId);
  inp.type = inp.type === 'password' ? 'text' : 'password';
}

function showErr(msg) {
  const b = document.getElementById('errBox');
  b.textContent = '⚠ ' + msg;
  b.className = 'err-box'; void b.offsetWidth; b.className = 'err-box on';
  document.getElementById('okBox').className = 'ok-box';
}
function showOk(msg) {
  const b = document.getElementById('okBox');
  b.textContent = '✓ ' + msg;
  b.className = 'ok-box on';
  document.getElementById('errBox').className = 'err-box';
}

document.getElementById('regForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = document.getElementById('fullName').value.trim();
  const phone = document.getElementById('phone').value.trim();
  const district = document.getElementById('district').value;
  const vdc = document.getElementById('vdc').value.trim();
  const ward = document.getElementById('ward').value.trim();
  const pwd = document.getElementById('pwd').value;
  const pwd2 = document.getElementById('pwd2').value;

  if (!name || !phone || !district) { showErr('कृपया अनिवार्य फिल्डहरू भर्नुहोस्।'); return; }
  if (pwd.length < 8) { showErr('पासवर्ड कम्तिमा ८ अक्षर हुनुपर्छ।'); return; }
  if (pwd !== pwd2) { showErr('पासवर्ड मेल खाएन। फेरि जाँच गर्नुहोस्।'); return; }

  const btn = document.getElementById('regBtn');
  btn.disabled = true;
  document.getElementById('btnTxt').textContent = 'दर्ता हुँदैछ…';
  document.getElementById('btnSpin').style.display = 'block';

  try {
    const resp = await fetch('/fchv/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, phone, district, vdc, ward, password: pwd })
    });
    const data = await resp.json();
    if (resp.ok && data.success) {
      document.getElementById('idPreview').textContent = data.fchv_id;
      showOk(`खाता बन्यो! ID: ${data.fchv_id} — लगइन पृष्ठमा जाँदैछ…`);
      document.getElementById('btnTxt').textContent = '✓ सफल!';
      setTimeout(() => { window.location.href = '/login'; }, 2000);
    } else {
      showErr(data.error || 'दर्ता असफल भयो।');
      btn.disabled = false;
      document.getElementById('btnTxt').textContent = 'खाता बनाउनुहोस्';
      document.getElementById('btnSpin').style.display = 'none';
    }
  } catch (err) {
    showErr('सर्भर त्रुटि: ' + err.message);
    btn.disabled = false;
    document.getElementById('btnTxt').textContent = 'खाता बनाउनुहोस्';
    document.getElementById('btnSpin').style.display = 'none';
  }
});

fetchCounter();
</script>
</body>
</html>
