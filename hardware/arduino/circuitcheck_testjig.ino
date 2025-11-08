/*
  CircuitCheck Test Jig Firmware
  
  This Arduino firmware controls a test jig for measuring electrical
  characteristics of electronic components like ICs and connectors.
  
  Features:
  - Multi-pin resistance measurement
  - Capacitance measurement
  - Leakage current detection
  - Basic timing measurements
  - Serial communication with PC
  
  Hardware Requirements:
  - Arduino Uno/Nano or ESP32
  - Multiplexer (CD74HC4067 or similar)
  - Current sensing resistors
  - Pogo pins for component contact
  - Optional: relay switches for isolation
  
  Author: CircuitCheck Team
  Version: 1.0
*/

#include <ArduinoJson.h>

// Pin definitions
#define MUX_S0 2
#define MUX_S1 3
#define MUX_S2 4
#define MUX_S3 5
#define MUX_SIG A0
#define MUX_EN 6

#define VOLTAGE_REF 5.0
#define ADC_RESOLUTION 1024

// Test parameters
#define NUM_TEST_PINS 4
#define MEASUREMENT_SAMPLES 10
#define DELAY_BETWEEN_SAMPLES 10

// Global variables
bool testInProgress = false;
String currentCommand = "";

void setup() {
  Serial.begin(115200);
  
  // Initialize multiplexer pins
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);
  pinMode(MUX_EN, OUTPUT);
  
  // Enable multiplexer
  digitalWrite(MUX_EN, LOW);
  
  // Initialize serial communication
  Serial.println("CircuitCheck Test Jig Ready");
  Serial.println("Commands: START_TEST, GET_STATUS, RESET");
}

void loop() {
  if (Serial.available()) {
    currentCommand = Serial.readStringUntil('\n');
    currentCommand.trim();
    
    if (currentCommand == "START_TEST") {
      performComponentTest();
    } else if (currentCommand == "GET_STATUS") {
      sendStatus();
    } else if (currentCommand == "RESET") {
      resetSystem();
    } else if (currentCommand == "PING") {
      Serial.println("PONG");
    } else {
      Serial.println("ERROR: Unknown command");
    }
  }
  
  delay(50);
}

void performComponentTest() {
  if (testInProgress) {
    Serial.println("ERROR: Test already in progress");
    return;
  }
  
  testInProgress = true;
  Serial.println("STATUS: Starting component test...");
  
  // Create JSON document for results
  DynamicJsonDocument doc(2048);
  doc["test_type"] = "electrical_analysis";
  doc["timestamp"] = millis();
  doc["status"] = "success";
  
  // Resistance measurements
  JsonObject resistance = doc.createNestedObject("resistance");
  resistance["pin1_pin2"] = measureResistance(0, 1);
  resistance["pin2_pin3"] = measureResistance(1, 2);
  resistance["pin3_pin4"] = measureResistance(2, 3);
  resistance["pin4_pin1"] = measureResistance(3, 0);
  
  delay(100);
  
  // Capacitance measurements (simplified)
  JsonObject capacitance = doc.createNestedObject("capacitance");
  capacitance["pin1_gnd"] = measureCapacitance(0);
  capacitance["pin2_gnd"] = measureCapacitance(1);
  capacitance["pin3_gnd"] = measureCapacitance(2);
  capacitance["pin4_gnd"] = measureCapacitance(3);
  
  delay(100);
  
  // Leakage current measurements
  JsonObject leakage = doc.createNestedObject("leakage_current");
  leakage["pin1"] = measureLeakageCurrent(0);
  leakage["pin2"] = measureLeakageCurrent(1);
  leakage["pin3"] = measureLeakageCurrent(2);
  leakage["pin4"] = measureLeakageCurrent(3);
  
  delay(100);
  
  // Basic timing measurements (placeholder)
  JsonObject timing = doc.createNestedObject("timing");
  timing["rise_time"] = measureRiseTime();
  timing["fall_time"] = measureFallTime();
  timing["propagation_delay"] = measurePropagationDelay();
  
  // Send results
  Serial.println("RESULT:");
  serializeJson(doc, Serial);
  Serial.println();
  
  testInProgress = false;
  Serial.println("STATUS: Test completed");
}

void selectMuxChannel(int channel) {
  digitalWrite(MUX_S0, channel & 0x01);
  digitalWrite(MUX_S1, (channel >> 1) & 0x01);
  digitalWrite(MUX_S2, (channel >> 2) & 0x01);
  digitalWrite(MUX_S3, (channel >> 3) & 0x01);
  delay(5); // Allow settling time
}

float measureResistance(int pin1, int pin2) {
  // Simple resistance measurement using voltage divider
  // This is a basic implementation - real implementation would be more sophisticated
  
  selectMuxChannel(pin1);
  float voltage1 = readAverageVoltage();
  
  selectMuxChannel(pin2);
  float voltage2 = readAverageVoltage();
  
  // Calculate resistance based on voltage difference
  float voltageDiff = abs(voltage1 - voltage2);
  
  if (voltageDiff < 0.1) {
    return 0.0; // Very low resistance (short circuit)
  } else if (voltageDiff > 4.5) {
    return 999999.0; // Very high resistance (open circuit)
  }
  
  // Simple calculation - in practice, this would use known reference resistors
  float resistance = (voltageDiff / (VOLTAGE_REF - voltageDiff)) * 1000.0; // Assume 1kΩ reference
  
  return resistance;
}

float measureCapacitance(int pin) {
  // Basic capacitance measurement using RC charging
  // This is a simplified implementation
  
  selectMuxChannel(pin);
  
  // Discharge capacitor
  pinMode(MUX_SIG, OUTPUT);
  digitalWrite(MUX_SIG, LOW);
  delay(10);
  
  // Start charging and measure time
  pinMode(MUX_SIG, INPUT);
  unsigned long startTime = micros();
  
  // Wait for voltage to reach threshold
  while (analogRead(MUX_SIG) < (ADC_RESOLUTION * 0.63)) { // ~63% of VCC
    if (micros() - startTime > 10000) break; // Timeout after 10ms
  }
  
  unsigned long chargeTime = micros() - startTime;
  
  // Calculate capacitance (simplified)
  // C = t / (R * ln(1/(1-V/Vcc)))
  float capacitance = chargeTime / 693.0; // Assuming 1kΩ resistor, simplified calculation
  
  return capacitance; // Returns in arbitrary units for this demo
}

float measureLeakageCurrent(int pin) {
  // Basic leakage current measurement
  selectMuxChannel(pin);
  
  // Apply voltage and measure current
  float voltage = readAverageVoltage();
  
  // Convert voltage to current using known sense resistor
  // This is a simplified calculation
  float current = voltage / 100000.0; // Assume 100kΩ sense resistor
  
  return current * 1000000.0; // Return in microamps
}

float measureRiseTime() {
  // Placeholder for rise time measurement
  // Real implementation would use fast sampling and edge detection
  return random(10, 50); // Return random value for demo (nanoseconds)
}

float measureFallTime() {
  // Placeholder for fall time measurement
  return random(8, 45); // Return random value for demo (nanoseconds)
}

float measurePropagationDelay() {
  // Placeholder for propagation delay measurement
  return random(15, 100); // Return random value for demo (nanoseconds)
}

float readAverageVoltage() {
  float sum = 0;
  
  for (int i = 0; i < MEASUREMENT_SAMPLES; i++) {
    sum += analogRead(MUX_SIG);
    delay(DELAY_BETWEEN_SAMPLES);
  }
  
  float average = sum / MEASUREMENT_SAMPLES;
  return (average / ADC_RESOLUTION) * VOLTAGE_REF;
}

void sendStatus() {
  DynamicJsonDocument doc(512);
  doc["status"] = testInProgress ? "testing" : "ready";
  doc["uptime"] = millis();
  doc["free_memory"] = getFreeMemory();
  doc["version"] = "1.0";
  
  Serial.println("STATUS:");
  serializeJson(doc, Serial);
  Serial.println();
}

void resetSystem() {
  testInProgress = false;
  
  // Reset all pins
  digitalWrite(MUX_S0, LOW);
  digitalWrite(MUX_S1, LOW);
  digitalWrite(MUX_S2, LOW);
  digitalWrite(MUX_S3, LOW);
  
  Serial.println("STATUS: System reset completed");
}

int getFreeMemory() {
  // Simple free memory calculation for Arduino
#ifdef __arm__
  // For ARM processors
  char top;
  return &top - reinterpret_cast<char*>(sbrk(0));
#elif defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_NANO)
  // For AVR processors
  extern int __heap_start, *__brkval;
  int v;
  return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
#else
  // Default fallback
  return -1;
#endif
}