//+------------------------------------------------------------------+
//|                                                     MT5_Signal_EA.mq5 |
//|                                                                  |
//|                                           Based on MQL5 samples   |
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//--- input parameters
input string InpSymbol = "EURUSD"; // Trading Symbol
input int InpSignalPeriod = 60; // Signal expiry period in seconds
input double InpLotSize = 0.01; // Lot size for trades (not used in this implementation)

//--- Global variables
CTrade trade;
CSymbolInfo symbol_info;
string filename = "signals.json";

//--- Indicator handle
int indicator_handle = 0;

//--- Buffers to store indicator values
double signal_buffer[];
double trend_buffer[];

//--- Previous bar info
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Set symbol info
   symbol_info.Name(InpSymbol);
   
   //--- Check if symbol is available
   if(!symbol_info.Select(InpSymbol))
   {
      Print("Symbol not available: ", InpSymbol);
      return(INIT_FAILED);
   }
   
   //--- Initialize indicator
   indicator_handle = iCustom(Symbol(), PERIOD_CURRENT, "MT5_Signal_Indicator", 
                              14, MODE_SMA, PRICE_CLOSE, 0.0001);
   
   if(indicator_handle == INVALID_HANDLE)
   {
      Print("Could not get indicator handle");
      return(INIT_FAILED);
   }
   
   //--- Initialize arrays
   ArraySetAsSeries(signal_buffer, true);
   ArraySetAsSeries(trend_buffer, true);
   
   Print("MT5 Signal EA initialized successfully");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("MT5 Signal EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check for new bar
   static datetime last_tick_time = 0;
   datetime current_time = TimeCurrent();
   
   // Process once per bar change
   if(last_tick_time != iTime(Symbol(), PERIOD_CURRENT, 0))
   {
      last_tick_time = iTime(Symbol(), PERIOD_CURRENT, 0);
      
      // Process signals
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
      Print("Error copying indicator buffers");
      return;
   }
   
   //--- Check for new signals on the current bar
   datetime current_bar_time = iTime(Symbol(), PERIOD_CURRENT, 0);
   
   //--- Only process if this is a new bar
   if(current_bar_time != last_bar_time)
   {
      last_bar_time = current_bar_time;
      
      //--- Check current bar for signal
      double current_signal = signal_buffer[0];
      double previous_signal = signal_buffer[1];
      
      //--- Only process if we have a new signal (different from previous bar)
      if(MathAbs(current_signal) > 0.5 && MathAbs(previous_signal) <= 0.5)
      {
         string direction = current_signal > 0 ? "CALL" : "PUT";
         
         // Write signal to file using the same function as the indicator
         WriteSignalToFile(InpSymbol, direction, TimeCurrent(), InpSignalPeriod);
      }
   }
}

//+------------------------------------------------------------------+
//| Write signal to JSON file                                        |
//+------------------------------------------------------------------+
void WriteSignalToFile(string symbol, string direction, datetime timestamp, int expirySeconds)
{
   // Create signal object
   string signalJson = "{";
   signalJson += "\"symbol\":\"" + symbol + "\",";
   signalJson += "\"direction\":\"" + direction + "\",";
   signalJson += "\"timestamp\":" + IntegerToString(timestamp) + ",";
   signalJson += "\"expiry_seconds\":" + IntegerToString(expirySeconds) + ",";
   signalJson += "\"confidence\":" + DoubleToString(0.7, 2) + ",";  // Fixed confidence
   signalJson += "\"strategy\":\"ma_crossover\"";
   signalJson += "}";
   
   // Read existing signals from file
   string signalsArray = "[]";
   int handle = FileOpen(filename, FILE_READ | FILE_TXT);
   if(handle != INVALID_HANDLE)
   {
      signalsArray = FileReadString(handle);
      FileClose(handle);
   }
   
   // Parse the existing array and add new signal
   string newSignalsArray = signalsArray;
   
   // If the array is empty or invalid, initialize it
   if(StringLen(signalsArray) < 3 || signalsArray[0] != '[')
   {
      newSignalsArray = "[" + signalJson + "]";
   }
   else
   {
      // Remove closing bracket
      newSignalsArray = StringSubstr(signalsArray, 0, StringLen(signalsArray) - 1);
      
      // Add comma if array is not empty
      if(StringLen(newSignalsArray) > 1)
         newSignalsArray += ",";
      
      // Add new signal and closing bracket
      newSignalsArray += signalJson + "]";
   }
   
   // Write the updated array back to file
   handle = FileOpen(filename, FILE_WRITE | FILE_TXT);
   if(handle != INVALID_HANDLE)
   {
      FileWriteString(handle, newSignalsArray);
      FileClose(handle);
      
      Print("Signal written to file: ", signalJson);
   }
   else
   {
      Print("Error opening file for writing: ", filename);
   }
}

//+------------------------------------------------------------------+
//| Expert advisor function called on new bar                        |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   // Handle chart events if needed
}