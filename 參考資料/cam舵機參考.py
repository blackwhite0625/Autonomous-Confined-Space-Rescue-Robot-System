from flask import Flask, render_template_string, request, jsonify
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device
import time

# 強制設定 gpiozero 使用 lgpio (樹莓派 5 必備)
Device.pin_factory = LGPIOFactory()

app = Flask(__name__)

# 初始化舵機 ( Pan: 水平 GPIO 24, Tilt: 垂直 GPIO 25 )
try:
    pan_servo = AngularServo(24, min_angle=-90, max_angle=90, min_pulse_width=0.0005, max_pulse_width=0.0025)
    tilt_servo = AngularServo(25, min_angle=-90, max_angle=90, min_pulse_width=0.0005, max_pulse_width=0.0025)
    
    # 初始歸零 (全局預設 X: -9 度, Y: 60 度)
    pan_servo.angle = -9
    tilt_servo.angle = 60
except Exception as e:
    print(f"初始化舵機失敗，請確認是否具備 GPIO 權限或腳位是否正確: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>樹莓派雲台操作</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.10.1/nipplejs.min.js"></script>
    <style>
        body { 
            margin: 0; padding: 0; 
            background-color: #1e1e1e; color: #ffffff; 
            display: flex; flex-direction: column; align-items: center; justify-content: center; 
            height: 100vh; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            user-select: none;
        }
        #joystick-container { 
            width: 300px; height: 300px; 
            position: relative; 
            background: #2a2a2a; 
            border-radius: 50%; 
            box-shadow: 0 0 30px rgba(0,0,0,0.8) inset, 0 0 10px rgba(255,255,255,0.1); 
            margin: 20px 0;
        }
        .status { color: #888; font-size: 0.9em; font-family: monospace; }
        h2 { margin-bottom: 5px; color: #4CAF50; }

        #patrol-btn {
            margin-top: 20px; padding: 10px 20px; font-size: 16px; font-weight: bold;
            background-color: #4CAF50; color: white; border: none;
            border-radius: 8px; cursor: pointer; transition: 0.3s;
        }
        #patrol-btn:hover { background-color: #45a049; }
        #patrol-btn.active { background-color: #f44336; }
        #patrol-btn.active:hover { background-color: #da190b; }
    </style>
</head>
<body>
    <h2>Raspberry Pi 5 雲台控制</h2>
    <div class="status">Pan (GPIO 24) | Tilt (GPIO 25)</div>
    
    <div id="joystick-container"></div>
    
    <div class="status" id="angle-display">X: -9° | Y: 60°</div>

    <button id="patrol-btn">自動巡邏</button>

    <script>
        var options = {
            zone: document.getElementById('joystick-container'),
            mode: 'static',
            position: {left: '50%', top: '50%'},
            color: '#4CAF50',
            size: 150
        };
        var manager = nipplejs.create(options);
        var display = document.getElementById('angle-display');
        var patrolBtn = document.getElementById('patrol-btn');

        let targetX = -9 / 90;
        let targetY = 60 / 90;
        
        let currentX = targetX;
        let currentY = targetY;
        
        let moveVectorX = 0;
        let moveVectorY = 0;
        
        let lastSentPan = null;
        let lastSentTilt = null;
        let isPatrolling = false;
        let patrolStartY = 60 / 90;

        // [優化] 用 AbortController 取代 isSending 鎖
        // 每次有新角度要送出時，先取消上一個還在飛的 fetch，立刻送新的
        // 這樣永遠不會因為等待舊回應而丟棄新角度，動作零跳格
        let currentController = null;

        function stopPatrol() {
            if (!isPatrolling) return;
            isPatrolling = false;
            patrolBtn.innerText = '自動巡邏';
            patrolBtn.classList.remove('active');
            moveVectorX = 0;
            moveVectorY = 0;
        }

        patrolBtn.addEventListener('click', function() {
            if (isPatrolling) {
                stopPatrol();
            } else {
                isPatrolling = true;
                patrolStartY = currentY;
                this.innerText = '停止巡邏';
                this.classList.add('active');
            }
        });

        manager.on('move', function (evt, data) {
            stopPatrol();
            moveVectorX = data.vector.x;
            moveVectorY = data.vector.y;
        });

        manager.on('end', function () {
            moveVectorX = 0;
            moveVectorY = 0;
        });

        setInterval(() => {
            if (isPatrolling) {
                targetX = (-9 / 90) + Math.sin(Date.now() * 0.0003) * 0.8;
                targetY = patrolStartY;
            } else {
                targetX += moveVectorX * 0.05;
                targetY += moveVectorY * 0.05;

                targetX = Math.max(-1, Math.min(1, targetX));
                targetY = Math.max(-1, Math.min(1, targetY));
            }

            // [優化] 手動模式用 0.35（跟手更緊），巡邏用 0.2（平穩擺動）
            let easingFactor = isPatrolling ? 0.2 : 0.35;
            currentX += (targetX - currentX) * easingFactor;
            currentY += (targetY - currentY) * easingFactor;

            let panAngle  = Math.round(currentX * 90);
            let tiltAngle = Math.round(currentY * 90);

            // [優化] 手動模式門檻 1 度（細膩），巡邏模式門檻 2 度（過濾抖動）
            let threshold = isPatrolling ? 2 : 1;

            let panChanged  = Math.abs(panAngle  - (lastSentPan  ?? panAngle  + 999)) >= threshold;
            let tiltChanged = Math.abs(tiltAngle - (lastSentTilt ?? tiltAngle + 999)) >= threshold;

            if (panChanged || tiltChanged) {
                lastSentPan  = panAngle;
                lastSentTilt = tiltAngle;
                display.innerText = `X: ${panAngle}° | Y: ${tiltAngle}°`;

                // [優化] 取消上一個 fetch，不等回應直接送新角度
                if (currentController) {
                    currentController.abort();
                }
                currentController = new AbortController();

                fetch('/move', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({pan: panAngle, tilt: tiltAngle}),
                    signal: currentController.signal
                })
                .then(() => { currentController = null; })
                .catch(err => {
                    // AbortError 是主動取消，不需要印出錯誤
                    if (err.name !== 'AbortError') {
                        console.error(err);
                    }
                    currentController = null;
                });
            }
        }, 40);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/move', methods=['POST'])
def move_servo():
    data = request.json
    if not data:
        return jsonify({"status": "error"}), 400
        
    pan_angle  = int(data.get('pan',  -9))
    tilt_angle = int(data.get('tilt', 60))

    pan_angle  = max(min(pan_angle,  90), -90)
    tilt_angle = max(min(tilt_angle, 90), -90)

    try:
        pan_servo.angle  = pan_angle
        tilt_servo.angle = tilt_angle
    except Exception as e:
        print(f"寫入舵機角度錯誤: {e}")

    return jsonify({"status": "success", "pan": pan_angle, "tilt": tilt_angle})

if __name__ == '__main__':
    print("伺服器已啟動！請在瀏覽器輸入 http://<樹莓派的IP位址>:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)