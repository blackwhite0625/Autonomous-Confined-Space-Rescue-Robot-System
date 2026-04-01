from flask import Flask, render_template_string, request, jsonify
from gpiozero import PWMOutputDevice, OutputDevice
import math
import time

# --- 初始化 Flask ---
app = Flask(__name__)

# --- GPIO 腳位配置 (根據使用者提供) ---
# L298N_FRONT (前輪板)
# FL: Left Front, FR: Right Front
# 腳位: ENA(12), IN1(6), IN2(5), IN3(19), IN4(13), ENB(26)
motor_fl_pwm = PWMOutputDevice(12)
motor_fl_in1 = OutputDevice(6)
motor_fl_in2 = OutputDevice(5)

motor_fr_pwm = PWMOutputDevice(26)
motor_fr_in1 = OutputDevice(19)
motor_fr_in2 = OutputDevice(13)

# L298N_REAR (後輪板)
# RL: Left Rear, RR: Right Rear
# 腳位: ENA(16), IN1(27), IN2(17), IN3(23), IN4(22), ENB(20)
motor_rl_pwm = PWMOutputDevice(16)
motor_rl_in1 = OutputDevice(27)
motor_rl_in2 = OutputDevice(17)

motor_rr_pwm = PWMOutputDevice(20)
motor_rr_in1 = OutputDevice(23)
motor_rr_in2 = OutputDevice(22)

def set_motor(pwm, in1, in2, speed):
    """控制單一馬達：速度範圍 -1.0 到 1.0"""
    if speed > 0.05:  # 正轉
        in1.on()
        in2.off()
        pwm.value = min(speed, 1.0)
    elif speed < -0.05:  # 反轉
        in1.off()
        in2.on()
        pwm.value = min(abs(speed), 1.0)
    else:  # 停止
        in1.off()
        in2.off()
        pwm.value = 0

def move_car(x, y, r):
    """
    麥克拉姆輪運動邏輯修正版
    x: 橫移 (-1 ~ 1)
    y: 前後 (-1 ~ 1)
    r: 旋轉 (-1 ~ 1)
    """
    
    # --- 單獨控制橫移與旋轉方向 ---
    # 根據測試結果，如果橫移或旋轉方向反了，直接在這裡加上負號
    # 目前設定：假設你需要反轉旋轉方向
    r = r 

    x = -x

    # 計算各輪速度 (最標準的麥克拉姆公式)
    speed_fl = y + x + r
    speed_fr = y - x - r
    speed_rl = y - x + r
    speed_rr = y + x - r

    # --- 硬體極性修正 ---
    # 根據測試，前輪馬達在該硬體配置下需反向輸出以與後輪同步
    speed_fl = -speed_fl
    speed_fr = -speed_fr

    # 歸一化處理，防止數值超過 1.0
    speeds = [abs(speed_fl), abs(speed_fr), abs(speed_rl), abs(speed_rr)]
    max_speed = max(speeds)
    if max_speed > 1.0:
        speed_fl /= max_speed
        speed_fr /= max_speed
        speed_rl /= max_speed
        speed_rr /= max_speed

    # 執行馬達控制
    set_motor(motor_fl_pwm, motor_fl_in1, motor_fl_in2, speed_fl)
    set_motor(motor_fr_pwm, motor_fr_in1, motor_fr_in2, speed_fr)
    set_motor(motor_rl_pwm, motor_rl_in1, motor_rl_in2, speed_rl)
    set_motor(motor_rr_pwm, motor_rr_in1, motor_rr_in2, speed_rr)

# --- HTML 模板 ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>RPi 5 Mecanum Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.10.1/nipplejs.min.js"></script>
    <style>
        body { 
            background: #1a1a1a; 
            color: #00f2ff; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 0; 
            overflow: hidden;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            height: 100vh;
        }
        h1 { margin-bottom: 20px; text-shadow: 0 0 10px #00f2ff; }
        #controls-container {
            display: flex; flex-wrap: wrap; justify-content: center; gap: 50px;
        }
        .joystick-zone {
            width: 250px; height: 250px;
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid #00f2ff; border-radius: 50%;
            position: relative; box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
        }
        .label { margin-top: 15px; font-weight: bold; font-size: 1.2rem; }
        #status { margin-top: 20px; color: #888; font-size: 0.9rem; }
    </style>
</head>
<body>
    <h1>MECANUM DRIVE</h1>
    
    <div id="controls-container">
        <div>
            <div id="move-joystick" class="joystick-zone"></div>
            <div class="label">移動 / 橫移</div>
        </div>
        <div>
            <div id="rotate-joystick" class="joystick-zone"></div>
            <div class="label">原地旋轉</div>
        </div>
    </div>

    <div id="status">等待連接...</div>

    <script>
        let moveData = { x: 0, y: 0 };
        let rotateData = { r: 0 };
        const statusEl = document.getElementById('status');

        const moveManager = nipplejs.create({
            zone: document.getElementById('move-joystick'),
            mode: 'static',
            position: {left: '50%', top: '50%'},
            color: '#00f2ff',
            size: 150
        });

        const rotateManager = nipplejs.create({
            zone: document.getElementById('rotate-joystick'),
            mode: 'static',
            position: {left: '50%', top: '50%'},
            color: '#ff007b',
            size: 150,
            lockX: true 
        });

        function sendCommand() {
            fetch('/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    x: moveData.x,
                    y: moveData.y,
                    r: rotateData.r
                })
            }).then(response => {
                if(response.ok) statusEl.innerText = "已連線 - 正常控制中";
            }).catch(err => {
                statusEl.innerText = "斷線中...";
            });
        }

        moveManager.on('move', (evt, data) => {
            const dist = data.distance / 75;
            moveData.x = Math.cos(data.angle.radian) * dist;
            moveData.y = Math.sin(data.angle.radian) * dist;
            sendCommand();
        });

        moveManager.on('end', () => {
            moveData.x = 0; moveData.y = 0;
            sendCommand();
        });

        rotateManager.on('move', (evt, data) => {
            rotateData.r = (data.instance.frontPosition.x / 75);
            sendCommand();
        });

        rotateManager.on('end', () => {
            rotateData.r = 0;
            sendCommand();
        });

        setInterval(() => {
            if (moveData.x !== 0 || moveData.y !== 0 || rotateData.r !== 0) {
                sendCommand();
            }
        }, 100);
    </script>
</body>
</html>
"""

# --- Flask 路由 ---
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/control', methods=['POST'])
def control():
    data = request.json
    x = data.get('x', 0)
    y = data.get('y', 0)
    r = data.get('r', 0)
    
    move_car(x, y, r)
    
    return jsonify(status="success")

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("正在關閉控制器...")
    finally:
        move_car(0, 0, 0)