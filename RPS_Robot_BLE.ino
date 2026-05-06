/*
 * RPS_Robot_BLE.ino
 * =================
 * ESP32 firmware for the RPS Robot arm.
 * Receives gesture commands over Bluetooth Low Energy (BLE)
 * using the Nordic UART Service (NUS) profile.
 *
 * This replaces the previous USB serial approach and works with:
 *   - The Python desktop app (ble_bridge.py)
 *   - The React Native mobile app (react-native-ble-plx)
 *   - Any NUS-compatible BLE terminal app for testing (nRF Connect etc.)
 *
 * Protocol (unchanged from serial version):
 *   Incoming: CMD|ROCK\n  CMD|PAPER\n  CMD|SCISSORS\n  CMD|PING\n
 *   Outgoing: ACK|ROCK\n  ACK|PAPER\n  ACK|PING\n  ERR|<msg>\n
 *
 * BLE Service UUIDs (Nordic UART Service):
 *   Service:  6E400001-B5A3-F393-E0A9-E50E24DCCA9E
 *   RX (app writes here):  6E400002-B5A3-F393-E0A9-E50E24DCCA9E
 *   TX (ESP32 notifies):   6E400003-B5A3-F393-E0A9-E50E24DCCA9E
 *
 * Hardware:
 *   - ESP32 (any variant with BLE)
 *   - Servo on pin SERVO_PIN (adjust for your wiring)
 *   - Status LED on pin LED_PIN (built-in LED on most boards = pin 2)
 *
 * Libraries required (install via Arduino Library Manager):
 *   - NimBLE-Arduino by h2zero (recommended - smaller, faster than ESP32 BLE)
 *     OR the built-in ESP32 BLE library (heavier but works)
 *   - ESP32Servo by Kevin Harrington
 *
 * To install NimBLE-Arduino:
 *   Arduino IDE -> Tools -> Manage Libraries -> search "NimBLE-Arduino"
 *
 * Board setup:
 *   Arduino IDE -> Tools -> Board -> ESP32 Arduino -> ESP32 Dev Module
 *   (or your specific ESP32 board variant)
 */

#include <NimBLEDevice.h>
#include <ESP32Servo.h>

// ── Pin configuration ─────────────────────────────────────────────────────────
#define SERVO_PIN   18      // Change to match your wiring
#define LED_PIN     2       // Built-in LED (active HIGH on most ESP32 boards)

// ── Servo positions (degrees) — adjust for your robot arm ────────────────────
#define POS_ROCK      10    // Fist
#define POS_PAPER     90    // Open hand
#define POS_SCISSORS  50    // Two fingers
#define POS_OPEN      90    // Rest/open position
#define POS_CLOSE     10    // Closed position

// ── BLE UUIDs (Nordic UART Service) ──────────────────────────────────────────
#define NUS_SERVICE_UUID  "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX_UUID       "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  // app writes
#define NUS_TX_UUID       "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  // esp32 notifies

#define DEVICE_NAME "RPS Robot"

// ── Globals ───────────────────────────────────────────────────────────────────
Servo robotServo;
NimBLEServer*         pServer         = nullptr;
NimBLECharacteristic* pTxCharacteristic = nullptr;
bool deviceConnected    = false;
bool oldDeviceConnected = false;

String inputBuffer = "";

// ── BLE Server Callbacks ──────────────────────────────────────────────────────
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer) override {
        deviceConnected = true;
        digitalWrite(LED_PIN, HIGH);
        Serial.println("[BLE] Client connected");
    }

    void onDisconnect(NimBLEServer* pServer) override {
        deviceConnected = false;
        digitalWrite(LED_PIN, LOW);
        Serial.println("[BLE] Client disconnected - restarting advertising");
        // Restart advertising so app can reconnect
        NimBLEDevice::startAdvertising();
    }
};

// ── BLE RX Characteristic Callbacks (data received from app) ─────────────────
class RxCallbacks : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic* pCharacteristic) override {
        std::string rxValue = pCharacteristic->getValue();
        if (rxValue.length() > 0) {
            // Append to buffer (handle chunked writes)
            for (char c : rxValue) {
                if (c == '\n') {
                    // Process complete command
                    processCommand(inputBuffer);
                    inputBuffer = "";
                } else {
                    inputBuffer += c;
                }
            }
        }
    }
};

// ── Command processing ────────────────────────────────────────────────────────
void processCommand(String cmd) {
    cmd.trim();
    Serial.print("[CMD] Received: ");
    Serial.println(cmd);

    String response = "";

    if (cmd == "CMD|ROCK") {
        moveServo(POS_ROCK);
        response = "ACK|ROCK";
    }
    else if (cmd == "CMD|PAPER") {
        moveServo(POS_PAPER);
        response = "ACK|PAPER";
    }
    else if (cmd == "CMD|SCISSORS") {
        moveServo(POS_SCISSORS);
        response = "ACK|SCISSORS";
    }
    else if (cmd == "CMD|OPEN") {
        moveServo(POS_OPEN);
        response = "ACK|OPEN";
    }
    else if (cmd == "CMD|CLOSE") {
        moveServo(POS_CLOSE);
        response = "ACK|CLOSE";
    }
    else if (cmd == "CMD|PING") {
        response = "ACK|PING";
    }
    else {
        response = "ERR|Unknown command: " + cmd;
        Serial.print("[ERR] Unknown: ");
        Serial.println(cmd);
    }

    // Send response back to app
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

// ── Servo movement ────────────────────────────────────────────────────────────
void moveServo(int targetDegrees) {
    Serial.print("[SERVO] Moving to ");
    Serial.println(targetDegrees);
    robotServo.write(targetDegrees);
    // Brief LED flash to confirm movement
    digitalWrite(LED_PIN, LOW);
    delay(50);
    digitalWrite(LED_PIN, deviceConnected ? HIGH : LOW);
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("[RPS Robot] Starting BLE...");

    // LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Servo
    ESP32PWM::allocateTimer(0);
    robotServo.setPeriodHertz(50);
    robotServo.attach(SERVO_PIN, 500, 2400);
    robotServo.write(POS_OPEN);  // Start in open/rest position
    delay(500);

    // BLE init
    NimBLEDevice::init(DEVICE_NAME);
    NimBLEDevice::setMTU(185);  // Increase MTU for larger messages

    // Create server
    pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    // Create Nordic UART Service
    NimBLEService* pService = pServer->createService(NUS_SERVICE_UUID);

    // TX characteristic (ESP32 -> App, notify)
    pTxCharacteristic = pService->createCharacteristic(
        NUS_TX_UUID,
        NIMBLE_PROPERTY::NOTIFY
    );

    // RX characteristic (App -> ESP32, write)
    NimBLECharacteristic* pRxCharacteristic = pService->createCharacteristic(
        NUS_RX_UUID,
        NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR
    );
    pRxCharacteristic->setCallbacks(new RxCallbacks());

    // Start service
    pService->start();

    // Start advertising
    NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(NUS_SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);  // Helps iPhone connections
    pAdvertising->setMaxPreferred(0x12);
    NimBLEDevice::startAdvertising();

    Serial.println("[RPS Robot] BLE advertising as '" DEVICE_NAME "'");
    Serial.println("[RPS Robot] Ready. Waiting for connection...");

    // Flash LED 3 times to indicate ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(200);
        digitalWrite(LED_PIN, LOW);
        delay(200);
    }
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    // Handle reconnection after disconnect
    if (!deviceConnected && oldDeviceConnected) {
        delay(500);
        NimBLEDevice::startAdvertising();
        Serial.println("[BLE] Restarted advertising");
        oldDeviceConnected = deviceConnected;
    }
    if (deviceConnected && !oldDeviceConnected) {
        oldDeviceConnected = deviceConnected;
    }

    // Slow LED pulse while waiting for connection
    if (!deviceConnected) {
        static unsigned long lastBlink = 0;
        if (millis() - lastBlink > 1000) {
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
            lastBlink = millis();
        }
    }

    delay(10);
}
