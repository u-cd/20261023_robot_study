// D9 / Servo1の、機構に取り付けていない予備サーボ専用。
// シリアル通信で90〜100を送り、改行で確定します。
#include <Servo.h>

Servo testServo;
char command[4];
byte length = 0;
bool invalid = false;

void setup() {
  Serial.begin(115200);
  testServo.write(90);  // 最初の指令を設定してから信号出力を開始
  testServo.attach(9);
  Serial.println("Ready: enter 90 to 100, then Enter.");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (length == 0 && !invalid) continue;

      int angle = 0;
      for (byte i = 0; i < length; i++) {
        angle = angle * 10 + (command[i] - '0');
      }

      if (!invalid && angle >= 90 && angle <= 100) {
        testServo.write(angle);
        Serial.print("Command accepted: ");
        Serial.println(angle);
      } else {
        Serial.println("Error: enter an integer from 90 to 100.");
      }

      length = 0;
      invalid = false;
    } else if (c >= '0' && c <= '9' && length < sizeof(command) - 1) {
      command[length++] = c;
    } else {
      invalid = true;
    }
  }
}
