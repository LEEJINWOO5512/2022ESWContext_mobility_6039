#include <Servo.h>

Servo myServo;
int servoPin =3;

void setup(){
  myServo.attach(servoPin);
}

void loop(){
  int angle;

  for (angle=0; angle<180;angle++){
    myServo.write(angle);
    delay(5);
  }
  for (angle=180;angle>0;angle--){
    myServo.write(angle);
    delay(5);
  }
}
