#define USE_ARDUINO_INTERRUPTS true    // Set-up low-level interrupts for most acurate BPM math.

#include <PulseSensorPlayground.h>     // Includes the PulseSensorPlayground Library.   

#include <SoftwareSerial.h>

 

//  Variables

const int PulseWire = 3;       // PulseSensor PURPLE WIRE connected to ANALOG PIN 0

int Threshold = 550;           // Determine which Signal to "count as a beat" and which to ignore.

                               // Use the "Gettting Started Project" to fine-tune Threshold Value beyond default setting.

                               // Otherwise leave the default "550" value. 
     
PulseSensorPlayground pulseSensor;  // Creates an instance of the PulseSensorPlayground object called "pulseSensor"

 

 

void setup() {   

  Serial.begin(9600);          // For Serial Monitor

  // Configure the PulseSensor object, by assigning our variables to it. 

  pulseSensor.analogInput(PulseWire);   

  pulseSensor.setThreshold(Threshold);   

  // Double-check the "pulseSensor" object was created and "began" seeing a signal. 

  pulseSensor.begin();

}


void loop() {

  int myBPM = pulseSensor.getBeatsPerMinute();  // Calls function on our pulseSensor object that returns BPM as an "int".
  int myBPM1 = myBPM/2;
  int myBPM2 = myBPM/1.5;
  
  if (myBPM <= 80) Serial.println("0");
  else if (myBPM >= 200) Serial.println(myBPM1);
  else if (150 < myBPM < 200) Serial.println(myBPM2);
  else Serial.println(myBPM); 

  delay(1000);                    // considered best practice in a simple sketch.

 

}
