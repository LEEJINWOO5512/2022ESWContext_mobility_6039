int motor1pin1 = 2;
int motor1pin2 = 3;

const int LED_PIN = 9;
const int brightness = 10;


void setup() {
  pinMode(motor1pin1, OUTPUT);
  pinMode(motor1pin2, OUTPUT);

  pinMode(LED_PIN, OUTPUT);

  Serial.begin(9600);
}

void loop() {

  //motor1pin1 active  
  digitalWrite(motor1pin1, HIGH);
  digitalWrite(motor1pin2, LOW);
  delay(5000);
 
  analogWrite( LED_PIN, brightness );
}
