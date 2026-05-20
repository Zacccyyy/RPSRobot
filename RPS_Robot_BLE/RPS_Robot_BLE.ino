/*
 * RPS_AmazingHand_BLE_PWM.ino
 * ============================
 * Startup menu: C = Command mode, D = Diagnostic mode
 *
 * Diagnostic flow:
 *   Select finger → W (whole) or A/B (individual servo)
 *   W mode: O = open, C = close (both servos together)
 *   A/B mode: jog individual servo, save open/close angles
 *
 * Finger pairs (A=left, B=right looking from back of hand):
 *   Thumb:  pin 15 (A) / pin 2  (B)
 *   Index:  pin 19 (A) / pin 4  (B)
 *   Middle: pin 17 (A) / pin 5  (B)
 *   Ring:   pin 18 (A) / pin 16 (B)
 */

#include <NimBLEDevice.h>
#include <ESP32Servo.h>

// ── Pin definitions ───────────────────────────────────────────────────────────
#define THUMB_A_PIN    15
#define THUMB_B_PIN    2
#define INDEX_A_PIN    19   // FIXED: was 0 (GPIO0 strapping pin conflict)
#define INDEX_B_PIN    4
#define MIDDLE_A_PIN   17   // FIXED: was 16
#define MIDDLE_B_PIN   5
#define RING_A_PIN     18   // FIXED: was 17
#define RING_B_PIN     16   // FIXED: was 18

// ── Servo objects ─────────────────────────────────────────────────────────────
Servo thumbA,  thumbB;
Servo indexA,  indexB;
Servo middleA, middleB;
Servo ringA,   ringB;

// ── Servo registry ────────────────────────────────────────────────────────────
// Index: 0=ThumbA, 1=ThumbB, 2=IndexA, 3=IndexB,
//        4=MiddleA, 5=MiddleB, 6=RingA, 7=RingB
const char* FINGER_NAMES[4] = { "Thumb", "Index", "Middle", "Ring" };

const char* SERVO_NAMES[8] = {
    "Thumb A  (pin 15)",
    "Thumb B  (pin 2) ",
    "Index A  (pin 19)",
    "Index B  (pin 4) ",
    "Middle A (pin 17)",
    "Middle B (pin 5) ",
    "Ring A   (pin 18)",
    "Ring B   (pin 16)"
};

int fingerA(int f) { return f * 2; }
int fingerB(int f) { return f * 2 + 1; }

// ── Calibrated angles ─────────────────────────────────────────────────────────
int openAngles[8]  = { 55, 125, 180,   0,  55, 125,  55, 125 };
int closeAngles[8] = {180,   0,  55,  90, 180,   0, 180,   0 };

// ── App mode ──────────────────────────────────────────────────────────────────
enum AppMode { MODE_MENU, MODE_COMMAND, MODE_DIAGNOSTIC };
AppMode currentMode = MODE_MENU;

// ── Diagnostic state ──────────────────────────────────────────────────────────
enum DiagState {
    DIAG_FINGER_SELECT,
    DIAG_SERVO_SELECT,
    DIAG_WHOLE_FINGER,
    DIAG_SINGLE_SERVO
};
DiagState diagState  = DIAG_FINGER_SELECT;
int       diagFinger = -1;
int       diagSide   = -1;
int       diagAngle  =  90;

// ── BLE ───────────────────────────────────────────────────────────────────────
#define NUS_SERVICE_UUID  "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX_UUID       "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX_UUID       "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#define DEVICE_NAME       "RPS Robot"

NimBLECharacteristic* pTxCharacteristic = nullptr;
bool   deviceConnected = false;
String bleBuffer       = "";
String serialBuffer    = "";

// ── Write to servo by index ───────────────────────────────────────────────────
void writeServo(int idx, int angle) {
    angle = constrain(angle, 0, 180);
    switch (idx) {
        case 0: thumbA.write(angle);  break;
        case 1: thumbB.write(angle);  break;
        case 2: indexA.write(angle);  break;
        case 3: indexB.write(angle);  break;
        case 4: middleA.write(angle); break;
        case 5: middleB.write(angle); break;
        case 6: ringA.write(angle);   break;
        case 7: ringB.write(angle);   break;
    }
}

// ── Apply saved angles ────────────────────────────────────────────────────────
void goNeutral() {
    for (int i = 0; i < 8; i++) writeServo(i, openAngles[i]);
}

void goClose() {
    for (int i = 0; i < 8; i++) writeServo(i, closeAngles[i]);
}

void openFinger(int f) {
    writeServo(fingerA(f), openAngles[fingerA(f)]);
    writeServo(fingerB(f), openAngles[fingerB(f)]);
}

void closeFinger(int f) {
    writeServo(fingerA(f), closeAngles[fingerA(f)]);
    writeServo(fingerB(f), closeAngles[fingerB(f)]);
}

// ── RPS Gestures ─────────────────────────────────────────────────────────────
void showPaper() {
    Serial.println("[SERVO] PAPER");
    goNeutral();
}

void showRock() {
    Serial.println("[SERVO] ROCK");
    goClose();
}

void showScissors() {
    Serial.println("[SERVO] SCISSORS");
    openFinger(1);
    openFinger(2);
    closeFinger(3);
    closeFinger(0);
}

void showOpen()  { Serial.println("[SERVO] OPEN");  goNeutral(); }
void showClose() { Serial.println("[SERVO] CLOSE"); goClose();   }

// ── Print helpers ─────────────────────────────────────────────────────────────
void printAllAngles() {
    Serial.println();
    Serial.println("═════════════════════════════════════");
    Serial.println("  Saved angles — copy into code:");
    Serial.println("═════════════════════════════════════");
    Serial.print("  int openAngles[8]  = {");
    for (int i = 0; i < 8; i++) {
        Serial.print(openAngles[i]);
        if (i < 7) Serial.print(", ");
    }
    Serial.println("};");
    Serial.print("  int closeAngles[8] = {");
    for (int i = 0; i < 8; i++) {
        Serial.print(closeAngles[i]);
        if (i < 7) Serial.print(", ");
    }
    Serial.println("};");
    Serial.println("═════════════════════════════════════");
}

// ── Menu display ──────────────────────────────────────────────────────────────
void showMainMenu() {
    Serial.println();
    Serial.println("═════════════════════════════════════");
    Serial.println("   RPS Robot — Startup Menu");
    Serial.println("═════════════════════════════════════");
    Serial.println("   C = Command mode  (ROCK/PAPER/SCISSORS)");
    Serial.println("   D = Diagnostic mode (finger/servo control)");
    Serial.println("═════════════════════════════════════");
    Serial.print  ("> ");
}

void showFingerSelect() {
    Serial.println();
    Serial.println("─────────────────────────────────────");
    Serial.println("  Diagnostic — Select a finger:");
    Serial.println("─────────────────────────────────────");
    Serial.println("  1 = Thumb");
    Serial.println("  2 = Index");
    Serial.println("  3 = Middle");
    Serial.println("  4 = Ring");
    Serial.println("─────────────────────────────────────");
    Serial.println("  P = Print all saved angles");
    Serial.println("  M = Back to main menu");
    Serial.println("─────────────────────────────────────");
    Serial.print  ("> ");
}

void showServoSelect() {
    Serial.println();
    Serial.println("─────────────────────────────────────");
    Serial.print  ("  Finger: "); Serial.println(FINGER_NAMES[diagFinger]);
    Serial.println("─────────────────────────────────────");
    Serial.println("  W = Whole finger (both servos)");
    Serial.print  ("  A = Servo A only — "); Serial.println(SERVO_NAMES[fingerA(diagFinger)]);
    Serial.print  ("  B = Servo B only — "); Serial.println(SERVO_NAMES[fingerB(diagFinger)]);
    Serial.println("─────────────────────────────────────");
    Serial.println("  X = Back to finger select");
    Serial.println("─────────────────────────────────────");
    Serial.print  ("> ");
}

void showWholeFingerControls() {
    Serial.println();
    Serial.println("─────────────────────────────────────");
    Serial.print  ("  Whole finger: "); Serial.println(FINGER_NAMES[diagFinger]);
    Serial.print  ("    A servo: "); Serial.println(SERVO_NAMES[fingerA(diagFinger)]);
    Serial.print  ("    B servo: "); Serial.println(SERVO_NAMES[fingerB(diagFinger)]);
    Serial.println("─────────────────────────────────────");
    Serial.println("  O = Open  (both to saved open angles)");
    Serial.println("  C = Close (both to saved close angles)");
    Serial.println("  X = Back to servo select");
    Serial.println("─────────────────────────────────────");
    Serial.print  ("> ");
}

void showSingleServoControls() {
    String sideName = (diagSide == 0) ? "A" : "B";
    int    idx      = (diagSide == 0) ? fingerA(diagFinger) : fingerB(diagFinger);
    Serial.println();
    Serial.println("─────────────────────────────────────");
    Serial.print  ("  Servo "); Serial.print(sideName);
    Serial.print  (" — "); Serial.println(SERVO_NAMES[idx]);
    Serial.print  ("  Current angle: "); Serial.println(diagAngle);
    Serial.print  ("  Saved open:    "); Serial.println(openAngles[idx]);
    Serial.print  ("  Saved close:   "); Serial.println(closeAngles[idx]);
    Serial.println("─────────────────────────────────────");
    Serial.println("  +  = +1°      -  = -1°");
    Serial.println("  .  = +10°     ,  = -10°");
    Serial.println("  O  = Jump to saved open angle");
    Serial.println("  A  = Save current as OPEN");
    Serial.println("  Z  = Save current as CLOSE");
    Serial.println("  X  = Back to servo select");
    Serial.println("─────────────────────────────────────");
    Serial.print  ("> ");
}

// ── Diagnostic input handlers ─────────────────────────────────────────────────
void handleDiagInput(char c) {

    if (diagState == DIAG_FINGER_SELECT) {
        if (c >= '1' && c <= '4') {
            diagFinger = c - '1';
            diagState  = DIAG_SERVO_SELECT;
            showServoSelect();
        }
        else if (c == 'P' || c == 'p') { printAllAngles(); showFingerSelect(); }
        else if (c == 'M' || c == 'm') { currentMode = MODE_MENU; showMainMenu(); }
        return;
    }

    if (diagState == DIAG_SERVO_SELECT) {
        if (c == 'W' || c == 'w') {
            diagState = DIAG_WHOLE_FINGER;
            openFinger(diagFinger);
            showWholeFingerControls();
        }
        else if (c == 'A' || c == 'a') {
            diagSide  = 0;
            diagAngle = openAngles[fingerA(diagFinger)];
            writeServo(fingerA(diagFinger), diagAngle);
            diagState = DIAG_SINGLE_SERVO;
            showSingleServoControls();
        }
        else if (c == 'B' || c == 'b') {
            diagSide  = 1;
            diagAngle = openAngles[fingerB(diagFinger)];
            writeServo(fingerB(diagFinger), diagAngle);
            diagState = DIAG_SINGLE_SERVO;
            showSingleServoControls();
        }
        else if (c == 'X' || c == 'x') {
            diagState = DIAG_FINGER_SELECT;
            showFingerSelect();
        }
        return;
    }

    if (diagState == DIAG_WHOLE_FINGER) {
        if (c == 'O' || c == 'o') {
            openFinger(diagFinger);
            Serial.print("  [OPEN]  "); Serial.println(FINGER_NAMES[diagFinger]);
            Serial.print("    A="); Serial.print(openAngles[fingerA(diagFinger)]);
            Serial.print("  B="); Serial.println(openAngles[fingerB(diagFinger)]);
        }
        else if (c == 'C' || c == 'c') {
            closeFinger(diagFinger);
            Serial.print("  [CLOSE] "); Serial.println(FINGER_NAMES[diagFinger]);
            Serial.print("    A="); Serial.print(closeAngles[fingerA(diagFinger)]);
            Serial.print("  B="); Serial.println(closeAngles[fingerB(diagFinger)]);
        }
        else if (c == 'X' || c == 'x') {
            diagState = DIAG_SERVO_SELECT;
            showServoSelect();
        }
        return;
    }

    if (diagState == DIAG_SINGLE_SERVO) {
        int idx = (diagSide == 0) ? fingerA(diagFinger) : fingerB(diagFinger);

        switch (c) {
            case '+':
                diagAngle = constrain(diagAngle + 1, 0, 180);
                writeServo(idx, diagAngle);
                Serial.print("  Angle: "); Serial.println(diagAngle);
                break;
            case '-':
                diagAngle = constrain(diagAngle - 1, 0, 180);
                writeServo(idx, diagAngle);
                Serial.print("  Angle: "); Serial.println(diagAngle);
                break;
            case '.':
                diagAngle = constrain(diagAngle + 10, 0, 180);
                writeServo(idx, diagAngle);
                Serial.print("  Angle: "); Serial.println(diagAngle);
                break;
            case ',':
                diagAngle = constrain(diagAngle - 10, 0, 180);
                writeServo(idx, diagAngle);
                Serial.print("  Angle: "); Serial.println(diagAngle);
                break;
            case 'O': case 'o':
                diagAngle = openAngles[idx];
                writeServo(idx, diagAngle);
                Serial.print("  Jumped to saved open: "); Serial.println(diagAngle);
                break;
            case 'A': case 'a':
                openAngles[idx] = diagAngle;
                Serial.print("  [SAVED] OPEN  "); Serial.print(SERVO_NAMES[idx]);
                Serial.print(" = "); Serial.println(diagAngle);
                break;
            case 'Z': case 'z':
                closeAngles[idx] = diagAngle;
                Serial.print("  [SAVED] CLOSE "); Serial.print(SERVO_NAMES[idx]);
                Serial.print(" = "); Serial.println(diagAngle);
                break;
            case 'X': case 'x':
                diagState = DIAG_SERVO_SELECT;
                showServoSelect();
                break;
        }
        return;
    }
}

// ── Main menu input ───────────────────────────────────────────────────────────
void handleMenuInput(char c) {
    if (c == 'C' || c == 'c') {
        currentMode = MODE_COMMAND;
        Serial.println();
        Serial.println("[MODE] Command mode — ROCK / PAPER / SCISSORS / OPEN / CLOSE");
        Serial.println("[MODE] Type M to return to menu");
        goNeutral();
    }
    else if (c == 'D' || c == 'd') {
        currentMode = MODE_DIAGNOSTIC;
        diagState   = DIAG_FINGER_SELECT;
        diagFinger  = -1;
        goNeutral();
        showFingerSelect();
    }
}

// ── Command processing ────────────────────────────────────────────────────────
void processCommand(String cmd) {
    cmd.trim();
    Serial.print("[CMD] "); Serial.println(cmd);
    String response = "";

    if      (cmd == "CMD|ROCK")     { showRock();     response = "ACK|ROCK"; }
    else if (cmd == "CMD|PAPER")    { showPaper();    response = "ACK|PAPER"; }
    else if (cmd == "CMD|SCISSORS") { showScissors(); response = "ACK|SCISSORS"; }
    else if (cmd == "CMD|OPEN")     { showOpen();     response = "ACK|OPEN"; }
    else if (cmd == "CMD|CLOSE")    { showClose();    response = "ACK|CLOSE"; }
    else if (cmd == "CMD|PING")     {                 response = "ACK|PING"; }
    else if (cmd == "CMD|RUDE")     { showRude();     response = "ACK|RUDE"; }
    else if (cmd == "CMD|HELP" || cmd == "CMD|M" || cmd == "CMD|MENU") {
        currentMode = MODE_MENU;
        showMainMenu();
    }
    else { response = "ERR|Unknown: " + cmd; }

    if (response.length() > 0 && deviceConnected) {
        String msg = response + "\n";
        pTxCharacteristic->setValue(msg.c_str());
        pTxCharacteristic->notify();
        Serial.print("[TX] "); Serial.println(msg);
    }
}

void showRude() {
    Serial.println("[SERVO] RUDE");
    openFinger(2);   // Middle open
    closeFinger(0);  // Thumb  close
    closeFinger(1);  // Index  close
    closeFinger(3);  // Ring   close
}

// ── BLE Callbacks ─────────────────────────────────────────────────────────────
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) {
        deviceConnected = true;
        Serial.println("[BLE] Client connected");
    }
    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) {
        deviceConnected = false;
        Serial.println("[BLE] Disconnected - restarting advertising");
        NimBLEDevice::startAdvertising();
    }
};

class RxCallbacks : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic* pCharacteristic, NimBLEConnInfo& connInfo) {
        std::string rxValue = pCharacteristic->getValue();
        for (char c : rxValue) {
            if (c == '\n') {
                if (bleBuffer.length() > 0) {
                    if (!bleBuffer.startsWith("CMD|")) bleBuffer = "CMD|" + bleBuffer;
                    bleBuffer.toUpperCase();
                    processCommand(bleBuffer);
                    bleBuffer = "";
                }
            } else {
                bleBuffer += c;
            }
        }
    }
};

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    thumbA.setPeriodHertz(50);  thumbA.attach(THUMB_A_PIN,   400, 2600);
    thumbB.setPeriodHertz(50);  thumbB.attach(THUMB_B_PIN,   400, 2600);
    indexA.setPeriodHertz(50);  indexA.attach(INDEX_A_PIN,   400, 2600);
    indexB.setPeriodHertz(50);  indexB.attach(INDEX_B_PIN,   400, 2600);
    middleA.setPeriodHertz(50); middleA.attach(MIDDLE_A_PIN, 400, 2600);
    middleB.setPeriodHertz(50); middleB.attach(MIDDLE_B_PIN, 400, 2600);
    ringA.setPeriodHertz(50);   ringA.attach(RING_A_PIN,     400, 2600);
    ringB.setPeriodHertz(50);   ringB.attach(RING_B_PIN,     400, 2600);

    goNeutral();
    delay(500);

    NimBLEDevice::init(DEVICE_NAME);
    NimBLEDevice::setMTU(185);
    NimBLEServer* pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());
    NimBLEService* pService = pServer->createService(NUS_SERVICE_UUID);
    pTxCharacteristic = pService->createCharacteristic(
        NUS_TX_UUID, NIMBLE_PROPERTY::NOTIFY);
    NimBLECharacteristic* pRxChar = pService->createCharacteristic(
        NUS_RX_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
    pRxChar->setCallbacks(new RxCallbacks());
    pService->start();
    NimBLEDevice::getAdvertising()->addServiceUUID(NUS_SERVICE_UUID);
    NimBLEDevice::startAdvertising();

    showMainMenu();
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r') continue;

        if (currentMode == MODE_MENU) {
            if (c != '\n') handleMenuInput(c);
        }
        else if (currentMode == MODE_DIAGNOSTIC) {
            if (c != '\n') handleDiagInput(c);
        }
        else if (currentMode == MODE_COMMAND) {
            if (c == '\n') {
                serialBuffer.trim();
                if (serialBuffer.length() > 0) {
                    if (!serialBuffer.startsWith("CMD|"))
                        serialBuffer = "CMD|" + serialBuffer;
                    serialBuffer.toUpperCase();
                    processCommand(serialBuffer);
                }
                serialBuffer = "";
            } else {
                serialBuffer += c;
            }
        }
    }
    delay(10);
}