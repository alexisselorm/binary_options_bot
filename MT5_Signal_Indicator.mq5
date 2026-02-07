//+------------------------------------------------------------------+
//|                                             MT5_Signal_Indicator.mq5 |
//|                                                                  |
//|                                           Based on MQL5 samples   |
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "1.00"

#include <Arrays\ArrayObj.mqh>

//--- indicator settings
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

//--- plot SignalBuffer
indicator_color1 clrGreen;
indicator_style1 STYLE_SOLID;
indicator_width1 1;
//--- plot TrendBuffer
indicator_color2 clrRed;
indicator_style2 STYLE_SOLID;
indicator_width2 1;

//--- indicator buffers
double SignalBuffer[];
double TrendBuffer[];

//--- input parameters
input int InpMAPeriod = 14; // MA Period
input ENUM_MA_METHOD InpMAMethod = MODE_SMA; // MA Method
input ENUM_APPLIED_PRICE InpAppliedPrice = PRICE_CLOSE; // Applied Price
input double InpDeviation = 0.0001; // Signal Deviation

//--- Global variables
string filename = "signals.json";
datetime lastBarTime = 0;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- indicator buffers mapping
   SetIndexBuffer(0,SignalBuffer,INDICATOR_DATA);
   SetIndexBuffer(1,TrendBuffer,INDICATOR_DATA);
   
   //--- set indicator properties
   ArraySetAsSeries(SignalBuffer,true);
   ArraySetAsSeries(TrendBuffer,true);
   
   //--- initialization done
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   //--- check for bars count
   if(rates_total < InpMAPeriod)
      return(0);

   //--- preliminary calculations
   int limit = rates_total - prev_calculated;
   if(prev_calculated > 0)
      limit++;

   //--- main calculation loop
   for(int i = 0; i < limit; i++)
   {
      // Calculate moving average
      double maValue = iMA(NULL, 0, InpMAPeriod, 0, InpMAMethod, InpAppliedPrice, i);
      
      // Determine trend direction
      if(i > 0)
      {
         if(close[i] > maValue && close[i-1] <= maValue)
         {
            // Uptrend detected - potential CALL signal
            SignalBuffer[i] = 1.0;  // CALL signal
            TrendBuffer[i] = 1.0;
         }
         else if(close[i] < maValue && close[i-1] >= maValue)
         {
            // Downtrend detected - potential PUT signal
            SignalBuffer[i] = -1.0; // PUT signal
            TrendBuffer[i] = -1.0;
         }
         else
         {
            SignalBuffer[i] = 0.0;
            TrendBuffer[i] = 0.0;
         }
      }
      else
      {
         SignalBuffer[i] = 0.0;
         TrendBuffer[i] = 0.0;
      }
   }
   
   // Check for new bar and new signal
   datetime currentBarTime = time[0];
   if(currentBarTime != lastBarTime)
   {
      lastBarTime = currentBarTime;
      
      // Check if we have a new signal on the current bar
      if(MathAbs(SignalBuffer[0]) > 0.5)  // We have a signal
      {
         string signalDirection = SignalBuffer[0] > 0 ? "CALL" : "PUT";
         
         // Write signal to file
         WriteSignalToFile(Symbol(), signalDirection, TimeCurrent(), 60);  // 60 seconds expiry
      }
   }
   
   return(rates_total);
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