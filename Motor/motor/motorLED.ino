
const int motor1pin1 = 2;
const int motor1pin2 = 3;
const int motor2pin1 = 4;
const int motor2pin2 = 5;

int LED_red = 9;
int LED_green = 10;

void setup() {
  pinMode(motor1pin1, OUTPUT);
  pinMode(motor1pin2, OUTPUT);
  pinMode(motor2pin1, OUTPUT);
  pinMode(motor2pin2, OUTPUT);

  pinMode(LED_red, OUTPUT);
  pinMode(LED_green, OUTPUT);
  
  Serial.begin(9600);
}

void loop() {
  
  if(Serial.available()>0){
    char c = Serial.read();
    Serial.print(c);

    if (c == '0') {
      analogWrite(motor1pin1, 255);
      analogWrite(motor1pin2, 0);
      analogWrite(motor2pin1, 255);
      analogWrite(motor2pin2, 0);

      digitalWrite(LED_red, LOW);
      digitalWrite(LED_green, HIGH);

    }
    else {
      analogWrite(motor1pin1, 200);
      analogWrite(motor1pin2, 195);
      analogWrite(motor2pin1, 200);
      analogWrite(motor2pin2, 195);

      digitalWrite(LED_red, HIGH);
      digitalWrite(LED_green, LOW);

    }
  }

  
}
