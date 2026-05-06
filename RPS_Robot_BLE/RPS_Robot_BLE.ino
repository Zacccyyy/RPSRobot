/*
 * RPS_Robot_BLE.ino
 * =================
 * ESP32 BLE firmware for the RPS Robot arm.
 * Controls 3 servos independently on pins 18, 19, 21.
 *
 * Servo mapping:
 *   Pin 18 = ROCK     (R)
 *   Pin 19 = PAPER    (P)
 *   Pin 21 = SCISSORS (S)
 *
 * Protocol:
 *   Incoming: CMD|ROCK\n  CMD|PAPER\n  CMD|SCISSORS\n  CMD|PING\n
 *             CMD|OPEN\n  CMD|CLOSE\n
 *   Outgoing: ACK|ROCK\n  ACK|PAPER\n  ACK|SCISSORS\n  ACK|PING\n
 *
 * Libraries needed (Tools -> Manage Libraries):
 *   - NimBLE-Arduino by h2zero
 *   - ESP32Servo by Kevin Harrington
 *
 * Board: ESP32 Dev Module
 */

#include <NimBLEDevice.h>
#include <ESP32Servo.h>

// ── Pin configuration ─────────────────────────────────────────────────────────
#define SERVO_ROCK_PIN      18
#define SERVO_PAPER_PIN     19
#define SERVO_SCISSORS_PIN  21
#define LED_PIN             2

// ── Servo positions — full extend or full retract ────────────────────────────
#define ACTIVE   180    // Full extend
#define REST       0    // Full retract

// ── Nordic UART Service UUIDs ─────────────────────────────────────────────────
#define NUS_SERVICE_UUID  "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX_UUID       "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX_UUID       "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#define DEVICE_NAME       "RPS Robot"

// ── Globals ───────────────────────────────────────────────────────────────────
Servo servoRock;
Servo servoPaper;
Servo servoScissors;

NimBLECharacteristic* pTxCharacteristic = nullptr;
bool   deviceConnected = false;
String inputBuffer     = "";

// ── Forward declarations ──────────────────────────────────────────────────────
void processCommand(String cmd);
void sendResponse(String msg);
void showRock();
void showPaper();
void showScissors();
void showOpen();
void showClose();

// ── BLE Server Callbacks ──────────────────────────────────────────────────────
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) {
        deviceConnected = true;
        digitalWrite(LED_PIN, HIGH);
        Serial.println("[BLE] Client connected");
    }

    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) {
        deviceConnected = false;
        digitalWrite(LED_PIN, LOW);
        Serial.println("[BLE] Client disconnected - restarting advertising");
        NimBLEDevice::startAdvertising();
    }
};

// ── RX Characteristic Callbacks ──────────────────────────────────────────────
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

// ── Servo control ─────────────────────────────────────────────────────────────

void showRock() {
    Serial.println("[SERVO] ROCK");
    servoRock.write(ACTIVE);
    servoPaper.write(REST);
    servoScissors.write(REST);
}

void showPaper() {
    Serial.println("[SERVO] PAPER");
    servoRock.write(REST);
    servoPaper.write(ACTIVE);
    servoScissors.write(REST);
}

void showScissors() {
    Serial.println("[SERVO] SCISSORS");
    servoRock.write(REST);
    servoPaper.write(REST);
    servoScissors.write(ACTIVE);
}

void showOpen() {
    // All servos to rest/neutral
    Serial.println("[SERVO] OPEN (all rest)");
    servoRock.write(REST);
    servoPaper.write(REST);
    servoScissors.write(REST);
}

void showClose() {
    // All servos to active
    Serial.println("[SERVO] CLOSE (all active)");
    servoRock.write(ACTIVE);
    servoPaper.write(ACTIVE);
    servoScissors.write(ACTIVE);
}

// ── Command processing ────────────────────────────────────────────────────────
void processCommand(String cmd) {
    cmd.trim();
    Serial.print("[CMD] ");
    Serial.println(cmd);

    String response = "";

    if      (cmd == "CMD|ROCK")     { showRock();     response = "ACK|ROCK"; }
    else if (cmd == "CMD|PAPER")    { showPaper();    response = "ACK|PAPER"; }
    else if (cmd == "CMD|SCISSORS") { showScissors(); response = "ACK|SCISSORS"; }
    else if (cmd == "CMD|OPEN")     { showOpen();     response = "ACK|OPEN"; }
    else if (cmd == "CMD|CLOSE")    { showClose();    response = "ACK|CLOSE"; }
    else if (cmd == "CMD|PING")     {                 response = "ACK|PING"; }
    else { response = "ERR|Unknown: " + cmd; }

    if (response.length() > 0 && deviceConnected) {
        sendResponse(response);
    }
}

void sendResponse(String msg) {
    msg += "\n";
    pTxCharacteristic->setValue(msg.c_str());
    pTxCharacteristic->notify();
    Serial.print("[TX] ");
    Serial.println(msg);
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("[RPS Robot] Starting...");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Allocate timers for servos
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);

    // Attach servos
    servoRock.setPeriodHertz(50);
    servoRock.attach(SERVO_ROCK_PIN, 500, 2400);

    servoPaper.setPeriodHertz(50);
    servoPaper.attach(SERVO_PAPER_PIN, 500, 2400);

    servoScissors.setPeriodHertz(50);
    servoScissors.attach(SERVO_SCISSORS_PIN, 500, 2400);

    // Start all servos at rest position
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

    NimBLEAdvertising* pAdv = NimBLEDevice::getAdvertising();
    pAdv->addServiceUUID(NUS_SERVICE_UUID);
    NimBLEDevice::startAdvertising();

    Serial.println("[RPS Robot] Advertising as '" DEVICE_NAME "' - ready");
    Serial.println("[RPS Robot] Servos: Rock=18, Paper=19, Scissors=21");

    // Flash 3 times = ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH); delay(200);
        digitalWrite(LED_PIN, LOW);  delay(200);
    }
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    // Slow blink while waiting for connection
    if (!deviceConnected) {
        static unsigned long lastBlink = 0;
        if (millis() - lastBlink > 1000) {
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
            lastBlink = millis();
        }
    }
    delay(10);
}
