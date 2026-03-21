//+------------------------------------------------------------------+
//|                                             MT5_Signal_Indicator.mq5 |
//|                                                                  |
//|                                   Production-Grade Signal Indicator|
//|                                   Only generates signals (EA writes)|
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "2.0.0"
#property strict

#include <Arrays\ArrayObj.mqh>

//--- Indicator settings
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

//--- Plot SignalBuffer
indicator_color1 clrGreen;
indicator_style1 STYLE_SOLID;
indicator_width1 1;
//--- Plot TrendBuffer
indicator_color2 clrRed;
indicator_style2 STYLE_SOLID;
indicator_width2 1;

//--- Indicator buffers
double SignalBuffer[];
double TrendBuffer[];

//--- Input parameters
input int InpMAPeriod = 14;            // MA Period
input ENUM_MA_METHOD InpMAMethod = MODE_SMA;  // MA Method
input ENUM_APPLIED_PRICE InpAppliedPrice = PRICE_CLOSE;  // Applied Price
input double InpDeviation = 0.0001;    // Signal Deviation
input int InpMinBarsBetweenSignals = 5;  // Minimum bars between signals

//--- Global variables
datetime lastBarTime = 0;
datetime lastSignalTime = 0;
int barsSinceLastSignal = 0;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Indicator buffers mapping
   SetIndexBuffer(0, SignalBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, TrendBuffer, INDICATOR_DATA);

   //--- Set indicator properties
   ArraySetAsSeries(SignalBuffer, true);
   ArraySetAsSeries(TrendBuffer, true);

   //--- Set empty plot labels
   PlotIndexSetString(0, PLOT_LABEL, "Signal");
   PlotIndexSetString(1, PLOT_LABEL, "Trend");

   Print("MT5 Signal Indicator initialized successfully");
   Print("MA Period: ", InpMAPeriod, " | Method: ", InpMAMethod);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("MT5 Signal Indicator deinitialized. Reason: ", reason);
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
   //--- Check for bars count
   if(rates_total < InpMAPeriod + 1)
      return(0);

   //--- Preliminary calculations
   int limit = rates_total - prev_calculated;
   if(prev_calculated > 0)
      limit++;

   //--- Main calculation loop
   for(int i = 0; i < limit; i++)
   {
      // Calculate moving average
      double maValue = iMA(NULL, 0, InpMAPeriod, 0, InpMAMethod, InpAppliedPrice, i);

      // Determine trend direction and signal strength
      if(i > 0)
      {
         if(close[i] > maValue && close[i-1] <= maValue)
         {
            // Uptrend detected - potential CALL signal
            SignalBuffer[i] = 1.0;  // CALL signal (strength = 1.0)
            TrendBuffer[i] = 1.0;
         }
         else if(close[i] < maValue && close[i-1] >= maValue)
         {
            // Downtrend detected - potential PUT signal
            SignalBuffer[i] = -1.0; // PUT signal (strength = -1.0)
            TrendBuffer[i] = -1.0;
         }
         else
         {
            // No crossover - neutral
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

   //--- Track bars since last signal for rate limiting
   datetime currentBarTime = time[0];
   if(currentBarTime != lastBarTime)
   {
      lastBarTime = currentBarTime;
      barsSinceLastSignal++;
   }

   //--- Note: Signal writing is handled by the EA, not the indicator
   //--- The indicator only sets buffer values for the EA to read
   
   return(rates_total);
}

//+------------------------------------------------------------------+
//| Check if a new signal is available (for EA to read)              |
//+------------------------------------------------------------------+
bool IsNewSignalAvailable(int &direction)
{
   // Check minimum bars between signals
   if(barsSinceLastSignal < InpMinBarsBetweenSignals)
   {
      return false;
   }
   
   // Check if we have a signal on current bar
   if(MathAbs(SignalBuffer[0]) > 0.5)
   {
      // Check if previous bar had no signal (new signal detection)
      if(MathAbs(SignalBuffer[1]) <= 0.5)
      {
         direction = (SignalBuffer[0] > 0) ? 1 : -1;
         barsSinceLastSignal = 0;
         lastSignalTime = TimeCurrent();
         return true;
      }
   }
   
   return false;
}
