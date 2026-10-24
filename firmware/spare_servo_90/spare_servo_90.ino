// D9 / Servo1に接続した、機構に取り付けていない予備サーボ専用。
// 基準角度を未確認の組立済みアームには使用しないでください。
#include <Servo.h>

Servo testServo;

void setup() {
  Serial.begin(115200);
  testServo.attach(9);
}

void loop() {
  testServo.write(90);
  Serial.println("Command: 90 degrees");
  delay(2000);

  testServo.write(100);
  Serial.println("Command: 100 degrees");
  delay(2000);
}
