// 5関節の手動テスト。角度は指令値であり、実測値ではありません。
// 関節1〜4は0〜180度、5は実機報告に基づく暫定50〜150度。
// 数値制限内でも干渉・電源不足・過負荷を検出できません。
// START直後の移動は現在角度不明のため速度制限できません。
#include <Servo.h>
#include <stdlib.h>
#include <string.h>

const byte JOINTS = 5;
const byte PINS[JOINTS] = {9, 6, 5, 3, 11};
const int MIN_ANGLES[JOINTS] = {0, 0, 0, 0, 50};
const int MAX_ANGLES[JOINTS] = {180, 180, 180, 180, 150};
int speedDps[JOINTS] = {25, 25, 25, 25, 25};  // 指令速度。実際の速度ではない。
Servo motors[JOINTS];
bool active[JOINTS] = {false, false, false, false, false};
int commanded[JOINTS] = {90, 90, 90, 90, 90};
int target[JOINTS] = {90, 90, 90, 90, 90};
unsigned long lastStep[JOINTS] = {0, 0, 0, 0, 0};
char line[48];
byte length = 0;
bool overflow = false;
int jogJoint = -1;
int jogDirection = 0;
unsigned long lastJog = 0;
const unsigned long JOG_TIMEOUT_MS = 250;

void help() {
  Serial.println(F("START joint angle | MOVE joint angle | STOP | OFF joint | STATUS"));
  Serial.println(F("joint: 1=base 2=shoulder 3=elbow 4=wrist 5=gripper"));
  Serial.println(F("Limits: joints 1..4=0..180 (not calibrated); gripper=50..150 (provisional)"));
  Serial.println(F("START may jump. Support arm before OFF. Check cable clearance."));
  Serial.println(F("STEP joint +|- : one degree; JOG joint +|- : hold (refresh within 250ms)"));
  Serial.println(F("SPEED joint degrees_per_second : 5..100; default 25; START is not limited"));
}

bool parseNumber(const char *text, long minimum, long maximum, int &value) {
  if (!text || !*text) return false;
  // 数字のみ受け付け、巨大な数値や負数は変換前に拒否する。
  if (strlen(text) > 3) return false;
  for (const char *p = text; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
  }
  long number = strtol(text, NULL, 10);
  if (number < minimum || number > maximum) return false;
  value = (int)number;
  return true;
}

void stopMotion() {
  jogJoint = -1;
  for (byte i = 0; i < JOINTS; ++i) target[i] = commanded[i];
  Serial.println(F("STOP: targets cleared; active joints still holding (not power OFF)"));
}

void processCommand() {
  line[length] = '\0';
  if (overflow) {
    Serial.println(F("ERROR: line too long"));
    return;
  }
  char *command = strtok(line, " \t");
  if (!command) return;
  char *jointText = strtok(NULL, " \t");
  char *angleText = strtok(NULL, " \t");
  char *extra = strtok(NULL, " \t");
  if (strcmp(command, "HELP") == 0 && !jointText) {
    help();
  } else if (strcmp(command, "STOP") == 0 && !jointText) {
    stopMotion();
  } else if (strcmp(command, "STATUS") == 0 && !jointText) {
    for (byte i = 0; i < JOINTS; ++i) {
      Serial.print(i + 1);
      Serial.print(active[i] ? " ON commanded=" : " OFF last-commanded=");
      Serial.print(commanded[i]);
      Serial.print(F(" target="));
      Serial.print(target[i]);
      Serial.print(F(" speed="));
      Serial.println(speedDps[i]);
      Serial.print(F("  limits="));
      Serial.print(MIN_ANGLES[i]);
      Serial.print(F(".."));
      Serial.println(MAX_ANGLES[i]);
    }
    Serial.println(F("Command values only; OFF joint physical angle is unknown."));
  } else {
    int joint, angle;
    if (!parseNumber(jointText, 1, JOINTS, joint)) {
      Serial.println(F("ERROR: joint must be 1..5"));
      return;
    }
    byte i = joint - 1;
    if (strcmp(command, "SPEED") == 0 && !extra && parseNumber(angleText, 5, 100, angle)) {
      // 速度変更では移動しない。対象関節の保留目標も解除する。
      if (jogJoint == i) jogJoint = -1;
      target[i] = commanded[i];
      speedDps[i] = angle;
      lastStep[i] = millis();
      Serial.println(F("SPEED accepted; motion target cleared for this joint"));
    } else if ((strcmp(command, "STEP") == 0 || strcmp(command, "JOG") == 0) &&
        angleText && !extra &&
        (strcmp(angleText, "+") == 0 || strcmp(angleText, "-") == 0)) {
      if (!active[i]) {
        Serial.println(F("ERROR: use START for this joint first"));
        return;
      }
      // 過去のMOVE目標を消し、現在の指令値からのみ操作する。
      for (byte j = 0; j < JOINTS; ++j) target[j] = commanded[j];
      jogJoint = -1;
      int direction = angleText[0] == '+' ? 1 : -1;
      if (strcmp(command, "STEP") == 0) {
        int next = commanded[i] + direction;
        if (next >= MIN_ANGLES[i] && next <= MAX_ANGLES[i] &&
            millis() - lastStep[i] >= (1000UL / speedDps[i])) {
          commanded[i] = target[i] = next;
          motors[i].write(next);
          lastStep[i] = millis();
        }
      } else {
        jogJoint = i;
        jogDirection = direction;
        lastJog = millis();
      }
      Serial.println(F("Manual command accepted"));
    } else if (strcmp(command, "OFF") == 0 && !angleText) {
      if (jogJoint == i) jogJoint = -1;
      motors[i].detach();
      active[i] = false;
      target[i] = commanded[i];
      Serial.println(F("Signal OFF; servo power remains connected. Support gravity-loaded joints."));
    } else if ((strcmp(command, "START") == 0 || strcmp(command, "MOVE") == 0) &&
               !extra && parseNumber(angleText, 0, 180, angle)) {
      if (angle < MIN_ANGLES[i] || angle > MAX_ANGLES[i]) {
        Serial.println(F("ERROR: outside joint limits; use STATUS"));
        return;
      }
      jogJoint = -1;
      if (strcmp(command, "START") == 0) {
        if (active[i]) {
          Serial.println(F("ERROR: already active; use MOVE"));
          return;
        }
        commanded[i] = target[i] = angle;
        motors[i].write(angle);
        motors[i].attach(PINS[i]);
        lastStep[i] = millis();
        active[i] = true;
        Serial.println(F("START accepted (initial movement is not speed-limited)"));
      } else {
        if (!active[i]) {
          Serial.println(F("ERROR: use START for this joint first"));
          return;
        }
        target[i] = angle;
        Serial.println(F("MOVE accepted: target set"));
      }
    } else {
      Serial.println(F("ERROR: invalid command or angle outside 0..180. Use HELP."));
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("ARM_TEST_V3"));
  Serial.println(F("READY: all servo signals OFF"));
  help();
}

void loop() {
  // 一度に処理する文字数を制限して、動作更新が止まらないようにする。
  for (byte count = 0; count < 64 && Serial.available() > 0; ++count) {
    char c = Serial.read();
    if (c == '\r' || c == '\n') {
      if (length || overflow) {
        processCommand();
        Serial.println(F("DONE"));
      }
      length = 0;
      overflow = false;
    } else if (length < sizeof(line) - 1) {
      line[length++] = c;
    } else {
      overflow = true;
    }
  }
  unsigned long now = millis();
  if (jogJoint >= 0 && now - lastJog >= JOG_TIMEOUT_MS) {
    target[jogJoint] = commanded[jogJoint];
    jogJoint = -1;
  }
  for (byte i = 0; i < JOINTS; ++i) {
    if (now - lastStep[i] >= (1000UL / speedDps[i])) {
      if (jogJoint == i) {
        int next = commanded[i] + jogDirection;
        if (next >= MIN_ANGLES[i] && next <= MAX_ANGLES[i]) {
          target[i] = next;
        }
      }
      if (active[i] && commanded[i] != target[i]) {
        commanded[i] += (target[i] > commanded[i]) ? 1 : -1;
        motors[i].write(commanded[i]);
        // STEPの待ち時間も実際に指令を変更した時刻から数える。
        // 停止中まで更新すると、受信したSTEPがほぼ常に無視される。
        lastStep[i] = now;
      }
    }
  }
}
