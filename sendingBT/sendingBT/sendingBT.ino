
#include <SoftwareSerial.h>

SoftwareSerial BtSerial(8, 9);
    
void setup() {
  Serial.begin(9600);
  BtSerial.begin(9600);
}

void loop() {
  if(Serial.available()>0){
    char a = Serial.read();
    Serial.println(a);
    BtSerial.println(a);    
  }
}
