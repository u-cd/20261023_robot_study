// USB通信の確認用。サーボへの信号は出力しません。
void setup() {
  Serial.begin(115200);
}

void loop() {
  Serial.println("Robot connection OK");
  delay(1000);
}
