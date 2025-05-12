import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from plot_utils import moving_avg, kalman_filter
from scipy.stats import norm, shapiro, probplot


@dataclass
class PlotConfig:
    show_speed_hist: bool = True
    show_hourly_vehicles: bool = True
    show_hourly_slow_cars: bool = True    
    show_hourly_people: bool = True
    show_hourly_rain: bool = True
    show_qq: bool = True
    show_fastest: bool = True
    show_people: bool = True    
    show_median: bool = True    
    PDT = ZoneInfo("America/Los_Angeles")


class RadarPlotter:
    def __init__(self, config: PlotConfig):
        self.config = config
        self.fig = None
        self.axes = {}
        
    def setup_plots(self):
        """Initialize the subplot layout based on enabled plots"""
        enabled_plots = sum([
            self.config.show_speed_hist,
            self.config.show_hourly_vehicles,
            self.config.show_hourly_slow_cars,
            self.config.show_hourly_people,
            self.config.show_hourly_rain,
            self.config.show_qq,
            self.config.show_fastest,
            self.config.show_median            
        ])
        
        rows = (enabled_plots + 1) // 2  # 2 plots per row
        cols = min(2, enabled_plots)
        
        self.fig = plt.figure(figsize=(15, 5*rows))
        current_pos = 1
        
        if self.config.show_speed_hist:
            self.axes['hist'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1

        if self.config.show_qq:
            # Assuming qq plot is a scatter plot of quantiles
            self.axes['qq'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1
        
        if self.config.show_hourly_vehicles:
            self.axes['hourly_vehicles'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1

        if self.config.show_hourly_slow_cars:
            self.axes['hourly_slow_cars'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1

        if self.config.show_hourly_people:
            self.axes['hourly_people'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1
        
        if self.config.show_hourly_rain:
            self.axes['hourly_rain'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1
                
        if self.config.show_fastest:
            self.axes['fastest'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1

        if self.config.show_median:
            self.axes['median'] = self.fig.add_subplot(rows, cols, current_pos)
            current_pos += 1

        # Add other plot spaces as needed
        
        # plt.tight_layout(pad=3.0)
        plt.tight_layout(
            h_pad=6.5,    # Increase vertical spacing between plots
            w_pad=2.0,    # Horizontal spacing between plots
            rect=[0.05, 0.05, 0.95, 0.95]  # Leave margins around the figure
        )
        

    # plot any of several event types as a histogram for each hour of the day
    def plot_hours(self, ax, hour_counts, s, label):
        ax.bar(range(24), hour_counts, color='skyblue', edgecolor='black')
        ax.set_xlabel("hour of day (PDT)", labelpad=1) # Added labelpad for spacing
        ax.set_ylabel("%s count" % label, labelpad=1)                
        prefix = ""
        if (label == "cars"):
            prefix = "Very Slow "        
        ax.set_title("%s%s per Hour  %s" % (prefix, label.capitalize(), s), y=1.0, pad=2)
        ax.set_xticks(range(24))  # Changed from ax.xticks to ax.set_xticks
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        total_events = hour_counts.sum()
        ax.text(
            0.98, 0.95, f"Total: {total_events}", 
            transform=ax.transAxes,  # Changed from plt.gca().transAxes
            ha='right', va='top', 
            fontsize=10, fontweight='normal',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.25')
        )

    def hour_count(self, ax, event_times, label):
        if not event_times:  # If no events
            hour_counts = np.zeros(24, dtype=int)  # Create array of zeros
            date_string = "No Data"
        else:
            # Convert to datetime and extract hours
            hours = np.array([datetime.fromtimestamp(ts, tz=self.config.PDT).hour for ts in event_times])
            # Count how many events fall into each hour
            hour_counts = np.bincount(hours, minlength=24)
            dt = datetime.fromtimestamp(event_times[0], tz=self.config.PDT)    
            date_string = dt.strftime("%a %#m/%#d/%y")

        self.plot_hours(ax, hour_counts, date_string, label)
        
        if len(event_times) > 0:  # Only calculate stats if we have events
            peak_hour = np.argmax(hour_counts)
            peak_count = hour_counts[peak_hour]
            total = hour_counts.sum()

        #print("Peak activity at hour %02d with %d %s (%.1f%% of total)" % 
        #    (peak_hour, peak_count, label, 100.0 * peak_count/total ))
            
    def plot_speed_histogram_OLD(self, dfg1: pd.DataFrame, firstDate: str, lastDate: str):
        """Plot speed histogram in its designated subplot"""
        if not self.config.show_speed_hist:
            return
            
        ax = self.axes['hist']
        data = dfg1['max'].to_numpy()
        
        # Your existing histogram plotting code here, but using ax instead of plt
        ax.hist(data, bins=12, range=(20, 80), edgecolor='black')
        ax.set_xlabel('km/h', labelpad=1)
        ax.set_ylabel('events', labelpad=1)
        ax.set_title('Vehicle Speeds', y=1.0, pad=2)
        ax.grid(True)
        
    def plot_hourly(self, events, label):
        """Plot hourly counts in its designated subplot"""
        ax = self.axes[("hourly_%s" % label)]

            # Handle either dict or DataFrame inputs
        times = []
        if isinstance(events, pd.DataFrame):
            times = events['start_time'].tolist()
        else:  # assume list of dicts
            times = [p["start_time"] for p in events]
        
        self.hour_count(ax, times, label)        

    def plot_fastest(self, speeds, dir, label):   
        ax = self.axes['fastest'] 
        ax.set_title("Fastest vehicle  %s" % label, y=1.0, pad=2)    
        ax.plot(speeds, 'x')
        cleaner = kalman_filter(speeds, dir)    
        smooth = moving_avg(cleaner, 7)
        ax.plot(smooth, linewidth = 1, color='#40B000')
        # plt.plot(cleaner, linewidth = 1, color='#B04000')
        ax.set_ylabel('speed, km/h', labelpad=1)
        ax.set_xlabel('sample number', labelpad=1)
        ax.grid('both')        

    def plot_median(self, speeds, dir, label):   
        ax = self.axes['median'] 
        ax.set_title("Median vehicle  %s" % label, y=1.0, pad=2)    
        ax.plot(speeds, 'x')
        cleaner = kalman_filter(speeds, dir)    
        smooth = moving_avg(cleaner, 7)
        ax.plot(smooth, linewidth = 1, color='#40B000')
        # plt.plot(cleaner, linewidth = 1, color='#B04000')
        ax.set_ylabel('speed, km/h', labelpad=1)
        ax.set_xlabel('sample number', labelpad=1)
        ax.grid('both')        


    def plot_qq(self, dfg1, dfRaw):
        ax = self.axes['qq']
        # Generate a QQ plot
        speeds = dfg1['max'].to_numpy()
        #stat, p = shapiro(speeds)
        #print("Shapiro-Wilk statistic = %.4f, p-value = %0.3e" % (stat, p))
        count = len(dfg1)
        firstDate = dfg1['datetime'].iloc[0][:-6] # only hh:mm
        epoch0 = dfRaw['epoch'].iloc[0]
        epoch1 = dfRaw['epoch'].iloc[-1]
        dur = (epoch1 - epoch0)/(60*60.0) # duration in hours
        probplot(speeds, dist="norm", plot=ax)
        title = ("Probability Plot   [%d in %.1f h] %s" % (count, dur, firstDate))
        ax.set_title(title, y=1.0, pad=2)
        ax.set_ylabel('speed, km/h', labelpad=1)
        ax.set_xlabel('theoretical quantiles', labelpad=1)
        ax.grid('both')        

    def plot_speed_histogram(self, dfg1, firstDate, lastDate, hr_string):
        """histogram plot of vehicle speeds."""        
        ax = self.axes['hist']
        pk_speeds = []
        bin_count = 12
        bin_range = (20, 80)
        bin_edges = np.linspace(bin_range[0], bin_range[1], bin_count + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_stats = []  # Store bin stats for final summary

        pk_speeds = dfg1['max'].tolist()
        index = len(dfg1)-1

        mu = np.mean(pk_speeds)
        sigma = np.std(pk_speeds)
        N = len(pk_speeds)

        # Histogram with counts
        counts, bins, patches = ax.hist(pk_speeds,
                                        bins=bin_edges,
                                        edgecolor='black',
                                        color='skyblue')

        # Scale y-axis so labels don't overflow top edge of graph
        max_count = max(counts)
        ax.set_ylim(0, max_count * 1.2)

        # Gaussian curve scaled to counts
        x = np.linspace(bin_range[0], bin_range[1], 500)
        y = norm.pdf(x, mu, sigma) * N * bin_width
        ax.plot(x, y, 'g--', label=f'Normal ($\mu$={mu:.1f}, $\sigma$={sigma:.1f})')

        bin_stats.clear()

        for i in range(len(bin_edges) - 1):
            p_bin = norm.cdf(bin_edges[i + 1], mu, sigma) - norm.cdf(bin_edges[i], mu, sigma)
            expected = N * p_bin
            observed = counts[i]

            if expected > 0:
                z = (observed - expected) / np.sqrt(expected)
                p_val = 2 * norm.sf(abs(z))
            else:
                p_val = 1.0

            # Save for summary
            bin_stats.append({
                'bin_range': f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}',
                'observed': int(observed),
                'expected': expected,
                'p_value': p_val
            })

            # Label formatting
            if p_val > 0.054:
                label = ""
            elif p_val > 0.01:
                label = f'{p_val:.2f}'
            else:
                label = f'{p_val:.1e}'

            # Highlight if surprising
            if p_val < 0.01:
                patches[i].set_edgecolor('red')
                patches[i].set_linewidth(2)

            ax.text((bin_edges[i] + bin_edges[i + 1]) / 2,
                    observed + 0.5,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=8)


        # ax.set_title(f'Speeds for {index+1} vehicles')
        ax.set_xlabel('speed, km/h', fontsize = 10, labelpad=1)
        ax.set_ylabel('vehicle count', fontsize = 10, labelpad=1)
        # Show x-axis ticks every 10 units
        xmin = bin_range[0]
        xmax = bin_range[1]+1
        step = 5
        ax.set_xticks(ticks=range(xmin, xmax, step))
        ax.tick_params(axis='y', labelsize=10)    
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(True, axis='y', linestyle=':', linewidth=1)
        ax.legend()
        firstDateStr = "Start: " +str(firstDate)[0:-3]
        ax.annotate(firstDateStr, 
                xy=(0, 1.01), xycoords='axes fraction',
                ha='left', va='bottom', fontsize=8)


        ax.annotate((hr_string + " ending " + lastDate[:-3]), 
                    xy=(1, 1.01), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8)

            # Final summary of most surprising bins
        print()
        bin_stats_sorted = sorted(bin_stats, key=lambda x: x['p_value'])
        leastP = bin_stats_sorted[0].get('p_value')
        if (leastP > 0.01):
            print("No histogram bins are that surprising- distribution looks normal.")
        else:
            print("Of %d bins, the most surprising:" % bin_count)

        for stat in bin_stats_sorted:
            if (stat['p_value'] > 0.01):
                continue
            print(f"Bin {stat['bin_range']}: Observed = {stat['observed']:3d}, "
                f"Expected = {stat['expected']:5.2e}, p-value = {stat['p_value']:.4f}")

    
    def show(self):
        """Display all plots"""
        plt.show()
        