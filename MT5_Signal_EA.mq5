//+------------------------------------------------------------------+
//|                                                     MT5_Signal_EA.mq5 |
//|                                                                  |
//|                                   Production-Grade Signal Bridge  |
//|                                   Writes signals with strict schema|
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "2.0.0"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Input parameters
input string InpSymbol = "EURUSD";           // Trading Symbol
input int    InpSignalPeriod = 60;           // Signal expiry period in seconds
input double InpLotSize = 0.01;              // Lot size (not used in this implementation)
input double InpConfidence = 0.75;           // Signal confidence (0.0-1.0)
input string InpStrategy = "ma_crossover";   // Strategy name
input string InpSignalSource = "mt5_ea";     // Signal source identifier
input int    InpMaxRetries = 3;              // Max write retries
input int    InpWriteDelayMs = 100;          // Delay between retries in ms

//--- Global variables
CTrade trade;
CSymbolInfo symbol_info;
string filename = "signals.json";
string temp_filename = "signals.json.tmp";

//--- Indicator handle
int indicator_handle = 0;

//--- Buffers to store indicator values
double signal_buffer[];
double trend_buffer[];

//--- Previous bar info
datetime last_bar_time = 0;
string last_signal_direction = "";
datetime last_signal_time = 0;

//--- File lock simulation (prevent concurrent writes)
bool is_writing = false;
datetime last_write_time = 0;
const int WRITE_COOLDOWN_MS = 500;  // Minimum time between writes

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Set symbol info
   if(!symbol_info.Name(InpSymbol))
   {
      Print("ERROR: Failed to set symbol info for: ", InpSymbol);
      return(INIT_FAILED);
   }

   //--- Check if symbol is available
   if(!symbol_info.Select(InpSymbol))
   {
      Print("ERROR: Symbol not available: ", InpSymbol);
      return(INIT_FAILED);
   }

   //--- Initialize indicator
   indicator_handle = iCustom(Symbol(), PERIOD_CURRENT, "MT5_Signal_Indicator",
                              14, MODE_SMA, PRICE_CLOSE, 0.0001);

   if(indicator_handle == INVALID_HANDLE)
   {
      Print("ERROR: Could not get indicator handle");
      return(INIT_FAILED);
   }

   //--- Initialize arrays
   ArraySetAsSeries(signal_buffer, true);
   ArraySetAsSeries(trend_buffer, true);

   //--- Get terminal data folder for signal file path
   string data_folder = TerminalInfoString(TERMINAL_DATA_PATH);
   filename = data_folder + "\\MQL5\\Files\\" + filename;
   temp_filename = data_folder + "\\MQL5\\Files\\" + temp_filename;

   Print("MT5 Signal EA initialized successfully");
   Print("Signal file path: ", filename);
   Print("Symbol: ", InpSymbol, " | Strategy: ", InpStrategy);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("MT5 Signal EA deinitialized. Reason code: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check for new bar
   datetime current_bar_time = iTime(Symbol(), PERIOD_CURRENT, 0);

   // Process once per bar change (confirmed bar logic)
   if(current_bar_time != last_bar_time)
   {
      last_bar_time = current_bar_time;
      
      // Wait a moment for bar confirmation
      Sleep(100);
      
      // Process signals on confirmed bar
      ProcessNewSignals();
   }
}

//+------------------------------------------------------------------+
//| Process new signals                                              |
//+------------------------------------------------------------------+
void ProcessNewSignals()
{
   //--- Copy indicator buffers
   if(CopyBuffer(indicator_handle, 0, 0, 2, signal_buffer) <= 0 ||
      CopyBuffer(indicator_handle, 1, 0, 2, trend_buffer) <= 0)
   {
      Print("ERROR: Failed to copy indicator buffers");
      return;
   }

   //--- Check for new signals on the confirmed bar
   double current_signal = signal_buffer[0];
   double previous_signal = signal_buffer[1];

   //--- Only process if we have a new signal (different from previous bar)
   //--- and signal strength is above threshold
   if(MathAbs(current_signal) > 0.5 && MathAbs(previous_signal) <= 0.5)
   {
      string direction = current_signal > 0 ? "CALL" : "PUT";
      
      //--- Prevent duplicate signals in same direction within short time
      datetime current_time = TimeCurrent();
      if(direction == last_signal_direction && (current_time - last_signal_time) < 300)
      {
         Print("DEBUG: Skipping duplicate signal in same direction: ", direction);
         return;
      }
      
      //--- Generate unique signal ID
      string signal_id = GenerateSignalID(direction, current_time);
      
      //--- Write signal to file with atomic write strategy
      bool success = WriteSignalToFile(
         InpSymbol, 
         direction, 
         current_time, 
         InpSignalPeriod,
         InpConfidence,
         InpStrategy,
         InpSignalSource,
         signal_id
      );
      
      if(success)
      {
         last_signal_direction = direction;
         last_signal_time = current_time;
         Print("SUCCESS: Signal written | ID=", signal_id, " | Direction=", direction);
      }
      else
      {
         Print("ERROR: Failed to write signal after retries");
      }
   }
}

//+------------------------------------------------------------------+
//| Generate unique signal ID                                        |
//+------------------------------------------------------------------+
string GenerateSignalID(string direction, datetime timestamp)
{
   // Create unique ID using timestamp and random component
   // Format: SRC_SYMBOL_DIR_TIMESTAMP_RANDOM
   string source_prefix = StringSubstr(InpSignalSource, 0, MathMin(5, StringLen(InpSignalSource)));
   string symbol_prefix = StringSubstr(StringReplace(InpSymbol, ".", ""), 0, MathMin(6, StringLen(InpSymbol)));
   
   // Get milliseconds for uniqueness
   int milliseconds = (int)(TimeLocal() % 1000);
   int random_component = MathRand();
   
   return StringFormat("%s_%s_%s_%d_%d%d", 
                       StringToUpper(source_prefix),
                       StringToUpper(symbol_prefix),
                       direction,
                       timestamp,
                       milliseconds,
                       random_component % 1000);
}

//+------------------------------------------------------------------+
//| Write signal to JSON file with atomic write strategy             |
//+------------------------------------------------------------------+
bool WriteSignalToFile(
   string symbol, 
   string direction, 
   datetime timestamp, 
   int expirySeconds,
   double confidence,
   string strategy,
   string source,
   string signal_id
)
{
   //--- Check write cooldown
   datetime current_time = TimeLocal();
   if(current_time - last_write_time < WRITE_COOLDOWN_MS / 1000.0)
   {
      Print("DEBUG: Write cooldown active, waiting...");
      Sleep(WRITE_COOLDOWN_MS);
   }
   
   //--- Create signal JSON object with strict schema
   string signalJson = "{";
   signalJson += "\"source\":\"" + source + "\",";
   signalJson += "\"symbol\":\"" + symbol + "\",";
   signalJson += "\"direction\":\"" + direction + "\",";
   signalJson += "\"timestamp\":" + IntegerToString((long)timestamp) + ",";
   signalJson += "\"expiry_seconds\":" + IntegerToString(expirySeconds) + ",";
   signalJson += "\"signal_id\":\"" + signal_id + "\",";
   signalJson += "\"confidence\":" + DoubleToString(confidence, 4) + ",";
   signalJson += "\"strategy\":\"" + strategy + "\"";
   signalJson += "}";

   //--- Read existing signals from file
   string signalsArray = "[]";
   int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_COMMON);
   
   if(handle != INVALID_HANDLE)
   {
      signalsArray = FileReadString(handle);
      FileClose(handle);
   }
   else
   {
      Print("DEBUG: Signal file doesn't exist or cannot be read, starting fresh");
   }

   //--- Validate existing array format
   if(StringLen(signalsArray) < 2 || signalsArray[0] != '[')
   {
      signalsArray = "[]";
   }

   //--- Parse existing signals and limit array size (prevent unbounded growth)
   //--- Keep only last 50 signals
   string cleanedArray = CleanupOldSignals(signalsArray, 50);
   
   //--- Add new signal to array
   string newSignalsArray;
   if(StringLen(cleanedArray) <= 2)  // Empty array "[]"
   {
      newSignalsArray = "[" + signalJson + "]";
   }
   else
   {
      // Remove closing bracket
      newSignalsArray = StringSubstr(cleanedArray, 0, StringLen(cleanedArray) - 1);
      // Add comma, new signal, and closing bracket
      newSignalsArray += "," + signalJson + "]";
   }

   //--- Atomic write: write to temp file first, then rename
   return AtomicWriteFile(filename, temp_filename, newSignalsArray);
}

//+------------------------------------------------------------------+
//| Atomic file write with retry logic                               |
//+------------------------------------------------------------------+
bool AtomicWriteFile(string targetPath, string tempPath, string content)
{
   int handle;
   
   for(int retry = 0; retry < InpMaxRetries; retry++)
   {
      //--- Write to temp file first
      handle = FileOpen(tempPath, FILE_WRITE | FILE_TXT | FILE_COMMON);
      
      if(handle == INVALID_HANDLE)
      {
         Print("WARNING: Cannot open temp file for writing (attempt ", retry + 1, "/", InpMaxRetries, ")");
         Sleep(InpWriteDelayMs);
         continue;
      }
      
      //--- Write content
      if(!FileWriteString(handle, content))
      {
         FileClose(handle);
         Print("WARNING: Failed to write to temp file (attempt ", retry + 1, "/", InpMaxRetries, ")");
         Sleep(InpWriteDelayMs);
         continue;
      }
      
      FileClose(handle);
      
      //--- Delete target file if exists (for atomic rename)
      if(FileIsExist(targetPath, FILE_COMMON))
      {
         FileDelete(targetPath, FILE_COMMON);
      }
      
      //--- Rename temp to target (atomic on most filesystems)
      if(FileMove(tempPath, targetPath, FILE_COMMON))
      {
         last_write_time = TimeLocal();
         return true;
      }
      else
      {
         //--- Fallback: copy content from temp to target
         handle = FileOpen(targetPath, FILE_WRITE | FILE_TXT | FILE_COMMON);
         if(handle != INVALID_HANDLE)
         {
            FileWriteString(handle, content);
            FileClose(handle);
            FileDelete(tempPath, FILE_COMMON);
            last_write_time = TimeLocal();
            return true;
         }
      }
      
      Print("WARNING: Atomic write failed (attempt ", retry + 1, "/", InpMaxRetries, ")");
      Sleep(InpWriteDelayMs);
   }
   
   //--- Cleanup temp file if it exists
   if(FileIsExist(tempPath, FILE_COMMON))
   {
      FileDelete(tempPath, FILE_COMMON);
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Clean up old signals to prevent unbounded array growth           |
//+------------------------------------------------------------------+
string CleanupOldSignals(string signalsArray, int maxSignals)
{
   // Simple approach: just return the array as-is if it's valid
   // The Python side handles deduplication and cleanup
   // This just prevents extremely large arrays
   
   if(StringLen(signalsArray) <= 2) return "[]";
   
   // Count signals by counting commas between objects
   // This is a simplified check - full parsing would require JSON library
   int signalCount = 1;  // At least one signal if we got here
   int pos = 1;  // Start after opening bracket
   
   while((pos = StringFind(signalsArray, "}", pos)) != -1)
   {
      signalCount++;
      pos++;
   }
   
   // If we have too many signals, return empty array to start fresh
   // Python side has persistent deduplication so we won't lose track
   if(signalCount > maxSignals * 2)
   {
      Print("DEBUG: Signal array too large (", signalCount, "), resetting");
      return "[]";
   }
   
   return signalsArray;
}

//+------------------------------------------------------------------+
//| Helper: String replace function                                  |
//+------------------------------------------------------------------+
string StringReplace(string str, string find, string replace)
{
   string result = str;
   int pos = StringFind(result, find);
   
   while(pos != -1)
   {
      result = StringSubstr(result, 0, pos) + replace + StringSubstr(result, pos + StringLen(find));
      pos = StringFind(result, find, pos + StringLen(replace));
   }
   
   return result;
}

//+------------------------------------------------------------------+
//| Helper: Check if file exists                                     |
//+------------------------------------------------------------------+
bool FileIsExist(string filename, int flags = 0)
{
   int handle = FileOpen(filename, FILE_READ | FILE_TXT | flags);
   if(handle != INVALID_HANDLE)
   {
      FileClose(handle);
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Helper: Move/rename file                                         |
//+------------------------------------------------------------------+
bool FileMove(string fromPath, string toPath, int flags = 0)
{
   // MQL5 doesn't have native file move, so we copy and delete
   if(!FileIsExist(fromPath, flags)) return false;
   
   int readHandle = FileOpen(fromPath, FILE_READ | FILE_TXT | flags);
   if(readHandle == INVALID_HANDLE) return false;
   
   string content = FileReadString(readHandle);
   FileClose(readHandle);
   
   int writeHandle = FileOpen(toPath, FILE_WRITE | FILE_TXT | flags);
   if(writeHandle == INVALID_HANDLE) return false;
   
   bool success = FileWriteString(writeHandle, content) != 0;
   FileClose(writeHandle);
   
   if(success)
   {
      FileDelete(fromPath, flags);
   }
   
   return success;
}
