/*
 * RPS_AmazingHand_BLE_PWM.ino
 * ============================
 * ESP32 BLE firmware controlling 8 PWM servos in antagonistic pairs,
 * mimicking AmazingHand finger geometry for RPS gestures.
 *
 * Finger pairs (Servo A / Servo B):
 *   Index:  pin 0  / pin 4
 *   Middle: pin 16 / pin 17
 *   Ring:   pin 5  / pin 18
 *   Thumb:  pin 19 / pin 21
 *
 * BLE Protocol (unchanged from original):
 *   Incoming: CMD|ROCK\n  CMD|PAPER\n  CMD|SCISSORS\n
 *             CMD|OPEN\n  CMD|CLOSE\n  CMD|PING\n
 *   Outgoing: ACK|ROCK\n  ACK|PAPER\n  ACK|SCISSORS\n etc.
 */

#include <NimBLEDevice.h>
#include <ESP32Servo.h>

// ── Pin definitions ───────────────────────────────────────────────────────────
#define INDEX_A_PIN    0
#define INDEX_B_PIN    4
#define MIDDLE_A_PIN   16
#define MIDDLE_B_PIN   17
#define RING_A_PIN     5
#define RING_B_PIN     18
#define THUMB_A_PIN    19
#define THUMB_B_PIN    21
#define LED_PIN        2

// ── Servo angle presets ───────────────────────────────────────────────────────
// Each finger pair: (A_angle, B_angle)
// Neutral / open hand
#define OPEN_A    55
#define OPEN_B   125

// Fully closed / fist
#define CLOSE_A  180
#define CLOSE_B    0

// Scissors — index & middle extended further than neutral
#define SCISSORS_A  25
#define SCISSORS_B 155

// Thumb close (slightly less aggressive than full close)
#define THUMB_CLOSE_A  160
#define THUMB_CLOSE_B   20

// ── Servo objects ─────────────────────────────────────────────────────────────
Servo indexA,  indexB;
Servo middleA, middleB;
Servo ringA,   ringB;
Servo thumbA,  thumbB;

// ── BLE config ────────────────────────────────────────────────────────────────
#define NUS_SERVICE_UUID  "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX_UUID       "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX_UUID       "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#define DEVICE_NAME       "RPS Robot"

NimBLECharacteristic* pTxCharacteristic = nullptr;
bool   deviceConnected = false;
String inputBuffer     = "";

// ── Finger helpers ────────────────────────────────────────────────────────────
void moveIndex (int a, int b) { indexA.write(a);  indexB.write(b);  }
void moveMiddle(int a, int b) { middleA.write(a); middleB.write(b); }
void moveRing  (int a, int b) { ringA.write(a);   ringB.write(b);   }
void moveThumb (int a, int b) { thumbA.write(a);  thumbB.write(b);  }

// ── RPS Gesture functions ─────────────────────────────────────────────────────

// PAPER — all fingers fully open
void showPaper() {
    Serial.println("[SERVO] PAPER");
    moveIndex (OPEN_A,  OPEN_B);
    moveMiddle(OPEN_A,  OPEN_B);
    moveRing  (OPEN_A,  OPEN_B);
    moveThumb (OPEN_A,  OPEN_B);
}

// ROCK — all fingers closed into fist
void showRock() {
    Serial.println("[SERVO] ROCK");
    // Open first to avoid servo strain
    showPaper();
    delay(300);
    moveIndex (CLOSE_A,      CLOSE_B);
    moveMiddle(CLOSE_A,      CLOSE_B);
    moveRing  (CLOSE_A,      CLOSE_B);
    moveThumb (THUMB_CLOSE_A, THUMB_CLOSE_B);
}

// SCISSORS — index + middle extended, ring + thumb closed
void showScissors() {
    Serial.println("[SERVO] SCISSORS");
    moveIndex (SCISSORS_A,   SCISSORS_B);
    moveMiddle(SCISSORS_A,   SCISSORS_B);
    moveRing  (CLOSE_A,      CLOSE_B);
    moveThumb (THUMB_CLOSE_A, THUMB_CLOSE_B);
}

// OPEN — all fingers to neutral
void showOpen() {
    Serial.println("[SERVO] OPEN");
    showPaper();
}

// CLOSE — all fingers fully closed
void showClose() {
    Serial.println("[SERVO] CLOSE");
    moveIndex (CLOSE_A,      CLOSE_B);
    moveMiddle(CLOSE_A,      CLOSE_B);
    moveRing  (CLOSE_A,      CLOSE_B);
    moveThumb (THUMB_CLOSE_A, THUMB_CLOSE_B);
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
    else { response = "ERR|Unknown: " + cmd; }

    if (response.length() > 0 && deviceConnected) {
        String msg = response + "\n";
        pTxCharacteristic->setValue(msg.c_str());
        pTxCharacteristic->notify();
        Serial.print("[TX] "); Serial.println(msg);
    }
}

// ── BLE Callbacks ─────────────────────────────────────────────────────────────
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) {
        deviceConnected = true;
        digitalWrite(LED_PIN, HIGH);
        Serial.println("[BLE] Client connected");
    }
    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) {
        deviceConnected = false;
        digitalWrite(LED_PIN, LOW);
        Serial.println("[BLE] Disconnected - restarting advertising");
        NimBLEDevice::startAdvertising();
    }
};

class RxCallbacks : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic* pCharacteristic, NimBLEConnInfo& connInfo) {
        std::string rxValue = pCharacteristic->getValue();
        for (char c : rxValue) {
            if (c == '\n') {
                processCommand(inputBuffer);
                inputBuffer = "";
            } else {
                inputBuffer += c;
            }
        }
    }
};

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("[RPS AmazingHand PWM] Starting...");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Allocate PWM timers
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    // Attach all 8 servos with standard 50Hz PWM range
    indexA.setPeriodHertz(50);  indexA.attach(INDEX_A_PIN,  500, 2400);
    indexB.setPeriodHertz(50);  indexB.attach(INDEX_B_PIN,  500, 2400);
    middleA.setPeriodHertz(50); middleA.attach(MIDDLE_A_PIN, 500, 2400);
    middleB.setPeriodHertz(50); middleB.attach(MIDDLE_B_PIN, 500, 2400);
    ringA.setPeriodHertz(50);   ringA.attach(RING_A_PIN,   500, 2400);
    ringB.setPeriodHertz(50);   ringB.attach(RING_B_PIN,   500, 2400);
    thumbA.setPeriodHertz(50);  thumbA.attach(THUMB_A_PIN,  500, 2400);
    thumbB.setPeriodHertz(50);  thumbB.attach(THUMB_B_PIN,  500, 2400);

    // Start in open/neutral position
    showOpen();
    delay(500);

    // BLE init
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

    Serial.println("[RPS AmazingHand PWM] Ready - advertising as '" DEVICE_NAME "'");

    // Flash 3x = ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH); delay(200);
        digitalWrite(LED_PIN, LOW);  delay(200);
    }
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    // ── Serial Monitor input for testing ──────────────────────────────────────
    if (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\n') {
            inputBuffer.trim();
            if (inputBuffer.length() > 0) {
                // Accept both short and full format
                // e.g. "ROCK" or "CMD|ROCK" both work
                if (!inputBuffer.startsWith("CMD|")) {
                    inputBuffer = "CMD|" + inputBuffer;
                }
                inputBuffer.toUpperCase();
                processCommand(inputBuffer);
            }
            inputBuffer = "";
        } else {
            inputBuffer += c;
        }
    }

    // ── BLE slow blink while waiting for connection ───────────────────────────
    if (!deviceConnected) {
        static unsigned long lastBlink = 0;
        if (millis() - lastBlink > 1000) {
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
            lastBlink = millis();
        }
    }
    delay(10);
}