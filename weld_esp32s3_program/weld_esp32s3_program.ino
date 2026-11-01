
#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// ==================================================
// LCD CONFIGURATION
// ==================================================
LiquidCrystal_I2C lcd(0x27, 16, 2);

const int LCD_SDA = 8;
const int LCD_SCL = 9;

unsigned long reportPageTimer = 0;
unsigned long idleTimer = 0;

bool reportRunning = false;
bool waitingForIdle = false;

int reportPage = 0;

int finalTotal = 36;
int finalNormal = 0;
int finalBurn = 0;
int finalSpatter = 0;
int finalOver = 0;
int finalReview = 0;

char finalResult[32] = "INSPECTION DONE";

// ==================================================
// MOTOR PINS - ORIGINAL SETTINGS
// ==================================================
const int PIN_STEP = 4;
const int PIN_DIR  = 5;

const int STEP_IDLE = HIGH;
const int STEP_ACTIVE = LOW;

const int DIR_POSITIVE = HIGH;
const int DIR_NEGATIVE = LOW;

// ==================================================
// MOTOR CONSTANTS - PRESERVED
// ==================================================
const long STEPS_PER_REV = 3200L;
const double STEP_DEG = 10.0;

const unsigned long PULSE_ACTIVE_US = 1500UL;
const unsigned long PULSE_IDLE_US = 1500UL;
const unsigned long DIR_SETUP_US = 2000UL;
const unsigned long SETTLE_MS = 500UL;

// ==================================================
// SERIAL
// ==================================================
char cmdBuf[160];
size_t cmdLen = 0;

// ==================================================
// MOTOR POSITION
// ==================================================
long curStep = 0;
double curAngle = 0.0;

// ==================================================
// FUNCTION DECLARATIONS
// ==================================================
long angleToStep(double angle);
void moveToStep(long target);
void settleAndReport();
void doStepCommand();
void doRotateCommand(const char *arg);
void handleCommand(char *cmd);

void lcdShow(const char *line1, const char *line2);
void lcdHandleCommand(char *arg);
void startFinalReport(char *arg);
void updateFinalReport();
void showReportPage(int page);

// ==================================================
// LCD DISPLAY FUNCTION
// ==================================================
void lcdShow(const char *line1, const char *line2) {

  lcd.clear();

  lcd.setCursor(0, 0);
  for (int i = 0; i < 16 && line1[i] != '\0'; i++) {
    lcd.print(line1[i]);
  }

  lcd.setCursor(0, 1);
  for (int i = 0; i < 16 && line2[i] != '\0'; i++) {
    lcd.print(line2[i]);
  }
}

// Python sends:
// LCD:ANGLE:120|IMAGE:13/36
void lcdHandleCommand(char *arg) {

  char *separator = strchr(arg, '|');

  if (separator == NULL) {
    lcdShow(arg, "");
    return;
  }

  *separator = '\0';

  lcdShow(arg, separator + 1);

  // A new status message cancels an old idle timer.
  waitingForIdle = false;
  reportRunning = false;
}

// ==================================================
// FINAL REPORT
// Python format:
// FINAL:36,30,2,3,1,0,DEFECT DETECTED
// ==================================================
void startFinalReport(char *arg) {

  int total, normal, burn, spatter, over, review;
  char result[32] = "INSPECTION DONE";

  int parsed = sscanf(
    arg,
    "%d,%d,%d,%d,%d,%d,%31[^\n]",
    &total,
    &normal,
    &burn,
    &spatter,
    &over,
    &review,
    result
  );

  if (parsed < 6) {
    Serial.println("ERROR:BAD_FINAL_REPORT");
    return;
  }

  finalTotal = total;
  finalNormal = normal;
  finalBurn = burn;
  finalSpatter = spatter;
  finalOver = over;
  finalReview = review;

  if (parsed >= 7) {
    strncpy(finalResult, result, sizeof(finalResult) - 1);
    finalResult[sizeof(finalResult) - 1] = '\0';
  }

  reportPage = 0;
  reportRunning = true;
  waitingForIdle = false;

  reportPageTimer = millis();

  showReportPage(reportPage);

  Serial.println("FINAL_REPORT_STARTED");
}

// ==================================================
// FINAL REPORT LCD PAGES
// ==================================================
void showReportPage(int page) {

  char line1[17];
  char line2[17];

  switch (page) {

    case 0:
      snprintf(line1, sizeof(line1), "INSPECTION DONE");
      snprintf(line2, sizeof(line2), "IMAGES: %02d", finalTotal);
      break;

    case 1:
      snprintf(line1, sizeof(line1), "NORMAL: %02d", finalNormal);
      snprintf(line2, sizeof(line2), "BURN: %02d SP:%02d",
               finalBurn, finalSpatter);
      break;

    case 2:
      snprintf(line1, sizeof(line1), "OVER WELD:%02d",
               finalOver);
      snprintf(line2, sizeof(line2), "REVIEW:%02d",
               finalReview);
      break;

    case 3:
      snprintf(line1, sizeof(line1), "FINAL RESULT");
      strncpy(line2, finalResult, 16);
      line2[16] = '\0';
      break;

    default:
      return;
  }

  lcdShow(line1, line2);
}

// ==================================================
// AUTOMATIC REPORT PAGE SWITCHING
// ==================================================
void updateFinalReport() {

  if (reportRunning) {

    if (millis() - reportPageTimer >= 3000UL) {

      reportPage++;

      if (reportPage < 4) {

        showReportPage(reportPage);
        reportPageTimer = millis();

      } else {

        reportRunning = false;
        waitingForIdle = true;

        // Two minutes after the final report.
        idleTimer = millis();

        Serial.println("FINAL_REPORT_COMPLETED");
      }
    }
  }

  if (waitingForIdle &&
      millis() - idleTimer >= 120000UL) {

    waitingForIdle = false;

    lcdShow("6G PIPE", "INSPECTION SYSTEM");

    Serial.println("LCD_IDLE_SCREEN");
  }
}

// ==================================================
// MOTOR ANGLE CALCULATION - ORIGINAL
// ==================================================
long angleToStep(double angle) {

  return (long)floor(
    angle * (double)STEPS_PER_REV / 360.0 + 0.5
  );
}

// ==================================================
// MOTOR MOVEMENT - ORIGINAL
// ==================================================
void moveToStep(long target) {

  long delta = target - curStep;

  if (delta > STEPS_PER_REV / 2)
    delta -= STEPS_PER_REV;

  if (delta < -STEPS_PER_REV / 2)
    delta += STEPS_PER_REV;

  if (delta != 0) {

    digitalWrite(
      PIN_DIR,
      delta > 0 ? DIR_POSITIVE : DIR_NEGATIVE
    );

    delayMicroseconds(DIR_SETUP_US);

    long n = labs(delta);

    for (long i = 0; i < n; i++) {

      digitalWrite(PIN_STEP, STEP_ACTIVE);
      delayMicroseconds(PULSE_ACTIVE_US);

      digitalWrite(PIN_STEP, STEP_IDLE);
      delayMicroseconds(PULSE_IDLE_US);

      if ((i & 0x3F) == 0x3F)
        yield();
    }
  }

  curStep =
    ((target % STEPS_PER_REV) + STEPS_PER_REV)
    % STEPS_PER_REV;
}

// ==================================================
// MOTOR REPLY - ORIGINAL
// ==================================================
void settleAndReport() {

  delay(SETTLE_MS);
  Serial.println("ANGLE_REACHED");
}

// ==================================================
// 10 DEGREE COMMAND - ORIGINAL
// ==================================================
void doStepCommand() {

  double newAngle = curAngle + STEP_DEG;

  moveToStep(angleToStep(newAngle));

  curAngle =
    (newAngle >= 360.0) ? newAngle - 360.0 : newAngle;

  settleAndReport();
}

// ==================================================
// ABSOLUTE ROTATION - ORIGINAL
// ==================================================
void doRotateCommand(const char *arg) {

  char *end = NULL;

  double a = strtod(arg, &end);

  if (end == arg) {
    Serial.println("ERROR:BAD_ANGLE");
    return;
  }

  while (*end == ' ' || *end == '\t')
    end++;

  if (*end != '\0') {
    Serial.println("ERROR:BAD_ANGLE");
    return;
  }

  if (!(a >= 0.0 && a <= 360.0)) {
    Serial.println("ERROR:RANGE");
    return;
  }

  moveToStep(angleToStep(a));

  curAngle = (a >= 360.0) ? 0.0 : a;

  settleAndReport();
}

// ==================================================
// SERIAL COMMAND HANDLER
// ==================================================
void handleCommand(char *cmd) {

  while (*cmd == ' ' || *cmd == '\t')
    cmd++;

  size_t n = strlen(cmd);

  while (n > 0 &&
         (cmd[n - 1] == ' ' || cmd[n - 1] == '\t')) {
    cmd[--n] = '\0';
  }

  for (size_t i = 0; i < n; i++) {
    cmd[i] = (char)toupper((unsigned char)cmd[i]);
  }

  if (n == 0)
    return;

  // Original motor commands
  if (strcmp(cmd, "1") == 0) {

    doStepCommand();

  } else if (strncmp(cmd, "ROTATE:", 7) == 0) {

    doRotateCommand(cmd + 7);

  } else if (strcmp(cmd, "ZERO") == 0) {

    curStep = 0;
    curAngle = 0.0;

    Serial.println("ZEROED");

  } else if (strcmp(cmd, "POS?") == 0) {

    Serial.print("POS:");
    Serial.print(curAngle, 3);
    Serial.print(" STEP:");
    Serial.println(curStep);

  } else if (strcmp(cmd, "PING") == 0) {

    Serial.println("PONG");

  // New LCD commands
  } else if (strncmp(cmd, "LCD:", 4) == 0) {

    lcdHandleCommand(cmd + 4);
    Serial.println("LCD_UPDATED");

  } else if (strncmp(cmd, "FINAL:", 6) == 0) {

    startFinalReport(cmd + 6);

  } else {

    Serial.println("ERROR:UNKNOWN_COMMAND");
  }
}

// ==================================================
// SETUP
// ==================================================
void setup() {

  Serial.begin(115200);

  // LCD initialization
  Wire.begin(LCD_SDA, LCD_SCL);

  lcd.init();
  lcd.backlight();

  lcdShow("6G PIPE INSPECT", "INITIALIZING...");

  // Motor initialization
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR, OUTPUT);

  digitalWrite(PIN_STEP, STEP_IDLE);
  digitalWrite(PIN_DIR, DIR_POSITIVE);

  delay(500);

  lcdShow("6G PIPE INSPECT", "SYSTEM READY");

  Serial.println("READY");
}

// ==================================================
// MAIN LOOP
// ==================================================
void loop() {

  // Update LCD report screens
  updateFinalReport();

  // Receive serial commands
  while (Serial.available() > 0) {

    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {

      if (cmdLen > 0) {

        cmdBuf[cmdLen] = '\0';
        cmdLen = 0;

        handleCommand(cmdBuf);
      }

    } else if (cmdLen < sizeof(cmdBuf) - 1) {

      cmdBuf[cmdLen++] = c;

    } else {

      cmdLen = 0;
    }
  }
}
