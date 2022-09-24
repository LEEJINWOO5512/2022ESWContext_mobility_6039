#include <Wire.h>
#include <SoftwareSerial.h>

SoftwareSerial btSerial(11,10);

void setup(){
  Serial.begin(115200);
  btSerial.begin(115200);
  Wire.begin();
  Serial.println("Heart rate sensor: ");
}

void loop(){
  Wire.requestFrom(0xA0 >> 1, 1);
  int accident = 100;
  while(Wire.available()){
    unsigned char c = Wire.read();
    Serial.print(c, DEC);
    Serial.print(",");
    Serial.println(accident);
    btSerial.print(c, DEC);
    btSerial.print(",");
    btSerial.println(accident);
  }
  delay(500);
}
