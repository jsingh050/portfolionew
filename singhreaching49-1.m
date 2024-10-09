%jaspreet singh
%march 25, 2024

%load the data, instantiate structure R with many fields that are
%behavioral data from the tasks the subjects completed
load("neuralData.mat")

%% Part 1 
% step 1: extract unique target locations from R 

TP=[R(3).timeTargetAcquire]
TP=[R().timeTargetAcquire]

%% step 2: find unique target locations 
TP=[R.TrialParams];

allTarget1X=[TP.target1X];
Target1X=unique(allTarget1X);

allTarget1Y=[TP.target1Y];
Target1Y=unique(allTarget1Y);

%% step 3: hand posistion for each trial
%timegocuephoto and timetargetscquire are fields instantiated by R and they
%hold the start and end times when reach bgins 
timegocue=[R.timeGoCuePHOTO]; 
timetargetacquire=[R.timeTargetAcquire];

%define trials
hhp=cell([1,length(R)]); %horozntal 
vhp=cell([1,length(R)]); %vertical 

%extract relevant trial data
for n=1:numel(R)
    hhp{n}=R(n).hhp(timegocue(n):timetargetacquire(n));
    vhp{n}=R(n).vhp(timegocue(n):timetargetacquire(n));
end

%% step 3 alternate
% Blue = [0 0 1]; % Blue
% Red = [1 0 0]; % Red
% Green = [0 1 0]; % Green
% Yellow = [1 1 0]; % Yellow
% Magenta = [1 0 1]; % Magenta
% Cyan = [0 1 1]; % Cyan
% Gray = [0.5 0.5 0.5]; % Gray
% Orange = [1 0.5 0]; % Orange
% Purple = [0.5 0 0.5]; % Purple
% Teal = [0 0.5 0.5]; % Teal
% 
% figure;
% set(groot,'defaultFigureColor','w');
% hold on
% for ch=1:length(R)
%     if allTarget1X(ch) == Target1X(8)
%         % Plot target 8 with gray color
%         plot(hhp{ch}, vhp{ch}, 'color', Gray)
%     else
%         % For other targets, plot with corresponding color
%         idx = find(Target1X == allTarget1X(ch));
%         if ~isempty(idx)
%             plot(hhp{ch}, vhp{ch}, 'color', colors(idx, :))
%         end
%     end
% end
% 
% % Plot markers for targets 1-8
% xt = [-64, -98, -86, -34, 34, -64, 86, 98]; % pull Target1X data
% yt = [-76, -17, 50, 93, 96, 50, -76]; % pull Target1Y data
% for ch = 1:7
%     plot(xt(ch), yt(ch), 'o', 'MarkerEdgeColor', [0, 0, 0], 'MarkerFaceColor', colors(ch, :))
% end
% 
% xlim([-120, 120])
% ylim([-90, 120])
% title('Hand Position', 'FontSize', 25, 'Color', 'b')
% xlabel('Horizontal Position in millimeters')
% ylabel('Vertical Position in Millimeters')
%%
% Define colors for the targets
myColorMap = [
    [0 0 1]; % Blue
    [1 0 0]; % Red
    [0 1 0]; % Green
    [1 1 0]; % Yellow
    [1 0 1]; % Magenta
    [0 1 1]; % Cyan
    [0.5 0.5 0.5]; % Gray
    [1 0.5 0]; % Orange
    [0.5 0 0.5]; % Purple
    [0 0.5 0.5]; % Teal
];

figure('Color', 'w'); % Create a new figure with white background
hold on

% Preallocate memory for hep and vep arrays
hep = cell(1, numel(R));
vep = cell(1, numel(R));
mean_hep = zeros(1, length(R));
mean_vep = zeros(1, length(R));

% Plot hand positions for each target
for n = 1:length(R)
    plot(hhp{n}, vhp{n}, 'color', myColorMap(Target1X == allTarget1X(n), :)); % Utilize logical indexing for color assignment
    hep{n} = R(n).hep(timegocue(n):timetargetacquire(n));
    vep{n} = R(n).vep(timegocue(n):timetargetacquire(n));
    mean_hep(n) = mean(hep{n});
    mean_vep(n) = mean(vep{n});
end

% Plot mean hand positions for each target
for target = 1:length(unique(Target1X))
    target_idx = Target1X == target;
    plot(mean_hep(target_idx), mean_vep(target_idx), 'o', 'MarkerEdgeColor', [0, 0, 1], 'MarkerFaceColor', myColorMap(target, :));
end

% Plot the target points
xt = [-64, -98, -86, -34, 34, -64, 86, 98]; % Pull Target1X data
yt = [-76, -17, 50, 93, 96, 50, -76, 17]; % Pull Target1Y data
scatter(xt, yt, 100, [0 0 1], 'x', 'filled'); % Use scatter for target points with colors from myColorMap

% Add the reference point
scatter(98, -17, 100, [0, 0, 1], 'o', 'filled'); % Reference point with blue color

% Set axis limits and add title, and axis labels
xlim([-100, 100]);
ylim([-100, 100]);
title('Hand Position', 'FontSize', 25, 'Color', 'b');
xlabel('Horizontal Position in millimeters');
ylabel('Vertical Position in Millimeters');


%% step 3 graph hand position for 8 targets and generate figure 1

Blue = [0 0 1]; % Blue
Red = [1 0 0]; % Red
Green = [0 1 0]; % Green
Yellow = [1 1 0]; % Yellow
Magenta = [1 0 1]; % Magenta
Cyan = [0 1 1]; % Cyan
Gray = [0.5 0.5 0.5]; % Gray
Orange = [1 0.5 0]; % Orange
Purple = [0.5 0 0.5]; % Purple
Teal = [0 0.5 0.5]; % Teal

figure;
set(groot,'defaultFigureColor','w');
hold on
for n=1:length(R)
% 1 
if allTarget1X(n)==Target1X(1)
plot(hhp{n},vhp{n},'color','r')
end

%2
if allTarget1X(n)==Target1X(2)
plot(hhp{n},vhp{n},'color','g')
end

%3
if allTarget1X(n)==Target1X(3)
plot(hhp{n},vhp{n},'color','c')
end

%4
if allTarget1X(n)==Target1X(4)
plot(hhp{n},vhp{n},'color','m')
end

%5
if allTarget1X(n)==Target1X(5)
plot(hhp{n},vhp{n},'color','y')
end

%6
if allTarget1X(n)==Target1X(6)
plot(hhp{n},vhp{n},'color','k')
end

%7
if allTarget1X(n)==Target1X(7)
plot(hhp{n},vhp{n},'color','b')
end

%8
if allTarget1X(n)==Target1X(8) 
    % if hhp{ch}<200
    plot(hhp{n},vhp{n},'color',[.5 .5 .5])
    % end
end
end

for n=1:7
xt=[-64, -98, -86, -34, 34, -64, 86, 98]; %pull Target1X data
yt=[-76, -17, 50, 93, 96, 50, -76]; %pull Target1Y data

plot(xt(n), yt(n), 'x', 'MarkerEdgeColor', [0, 0, 1], 'MarkerFaceColor', 'b', 'MarkerSize', 22, 'LineWidth', 2)
text(98, -17, num2str(8), 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')


plot(98,-17,'o','MarkerEdgeColor','g','MarkerFaceColor',[0,0,1])

hep=cell([1,length(R)]);
vep=cell([1,length(R)]);
mean_hep=zeros([1,length(R)]);
mean_vep=zeros([1,length(R)]);
for n=1:length(R)
    hep{n}=R(n).hep(timegocue(n):timetargetacquire(n));
    vep{n}=R(n).vep(timegocue(n):timetargetacquire(n));

     %calculate mean
    mean_hep(n)=mean(hep{n});
    mean_vep(n)=mean(vep{n});
end

hold on
for n=1:length(R)
%target 1
if allTarget1X(n)==Target1X(1)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor','r')
end

%target 2
if allTarget1X(n)==Target1X(2)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.8500 0.3250 0.0980])
end

%target 3
if allTarget1X(n)==Target1X(3)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.9290 0.6940 0.1250])
end

%target 4
if allTarget1X(n)==Target1X(4)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.4940 0.1840 0.5560])
end

%target 5
if allTarget1X(n)==Target1X(5)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.4660 0.6740 0.1880])
end

%target 6
if allTarget1X(n)==Target1X(6)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.3010 0.7450 0.9330])
end

%target 7
if allTarget1X(n)==Target1X(7)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.6350 0.0780 0.1840])
end

%target 8
if allTarget1X(n)==Target1X(8)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[1 1 1])
end
end
xlim([-200,200])
ylim([-200,200])

title('Hand Position', 'FontSize', 25, 'Color', 'k')
xlabel('Horizonal Position in millimeters')
ylabel('Vertical Position in Millimeters')
% Add legend for colors
legend('Target 1', 'Target 2', 'Target 3', 'Target 4', 'Target 5', 'Target 6', 'Target 7', 'Target 8', 'Location', 'best')
end
%%
figure;
set(groot,'defaultFigureColor','w');
hold on

% Define colors for targets
colors = {[1 0 0], [0 1 0], [0 1 1], [1 0 1], [1 1 0], [0.5 0.5 0.5], [0 0 1], [0 0 0]};

% Plot hand position for each trial
for n = 1:length(R)
    for targetIndex = 1:length(Target1X)
        if allTarget1X(n) == Target1X(targetIndex)
            plot(hhp{n}, vhp{n}, 'Color', colors{targetIndex});
        end
    end
end

% Plot eye position targets
for n = 1:7
    xt = [-64, -98, -86, -34, 34, -64, 86, 98];
    yt = [-76, -17, 50, 93, 96, 50, -76];
    plot(xt(n), yt(n), 'x', 'MarkerEdgeColor', [0, 0, 1], 'MarkerFaceColor', 'b', 'MarkerSize', 22, 'LineWidth', 2)
end

% Plot the last eye position target with a different marker
plot(98, -17, 'o', 'MarkerEdgeColor', 'g', 'MarkerFaceColor', [0, 0, 1]);

% Plot mean eye position for each target
for n = 1:length(R)
    for targetIndex = 1:length(Target1X)
        if allTarget1X(n) == Target1X(targetIndex)
            plot(mean_hep(n), mean_vep(n), 'o', 'MarkerEdgeColor', [0, 0, 1], 'MarkerFaceColor', colors{targetIndex});
        end
    end
end

% Set axis limits
xlim([-200, 200]);
ylim([-200, 200]);

% Title and labels
title('Hand and Eye Position', 'FontSize', 25, 'Color', 'k');
xlabel('Horizontal Position (mm)');
ylabel('Vertical Position (mm)');

% Add legend
legend('Target 1', 'Target 2', 'Target 3', 'Target 4', 'Target 5', 'Target 6', 'Target 7', 'Target 8', 'Location', 'best');


%% step 5 
%calculate velocity(x/t= velocity)
velocity=cell([1,numel(R)]);
for n=1:numel(R)
    for j=1:numel(hhp{n})-1
    velocity{n}(j)=((hhp{n}(j+1)-hhp{n}(j))^2+(vhp{n}(j+1)-vhp{n}(j))^2)^(1/2);
    end
end

%set threshold
threshold=.15; %somewhere between 15-25 percent
% %find max speed and reaction time for each trial
% peak_velocity=zeros([1,length(R)]);
% threshold_velocity=zeros([1,length(R)]);
% react_time=zeros([1,length(R)]);
% for ch=1:length(R)
%     [peak_velocity(ch),id]=max(velocity{ch});
%     threshold_velocity(ch)=thresh*peak_velocity(ch);
%     react_time(ch)=find(velocity{ch}(1:id)==interp1(velocity{ch}(1:id),velocity{ch}(1:id),threshold_velocity(ch),'nearest'));
% end
% Preallocate arrays
peak_velocity = zeros(1, numel(R));
threshold_velocity = zeros(1, numel(R));
react_time = zeros(1, numel(R));

% Calculate peak velocity and reaction time
for n = 1:numel(R)
    % Find peak velocity and its index
    [peak_velocity(n), idx] = max(velocity{n});
    
    % Calculate threshold velocity
    threshold_velocity(n) = threshold * peak_velocity(n);
    
    % Find reaction time using interpolation
    react_time(n) = find(velocity{n}(1:idx) >= threshold_velocity(n), 1, 'first');
end

%plot speed for 8 trials, visualize threshold
figure;
hold on
for n=1:8
    subplot(2,4,n)
    plot(velocity{n})
    yline(threshold_velocity(n))
    xline(react_time(n))
    xlabel('Time in milliseconds')
    ylabel('Speed in mm/ms')
end
% title 'Change in Speed over Time over 9 Trials'

%calculate mean and standard deviation reaction time
mean_time=mean(react_time);
standev_time=std(react_time);
[h,p]=ttest(react_time,allTarget1X);

%calculate mean and standard deviation rt for each target
mean_rt_t=zeros([1,8]); %initialize array
std_rt_t=zeros([1,8]);
for n=1:length(Target1X)
    mean_rt_t(n)=mean(react_time(allTarget1X==Target1X(n)));
    std_rt_t(n)=std(react_time(allTarget1X==Target1X(n)));
end

%% step 7: computer refresh leads to latency or lag time stored as timegocuephoto
%mean and std latency
meanlatency=mean(timegocue-[R.timeGoCue]);
stdlatency=std(timegocue-[R.timeGoCue]);
%%
% Define the data
mean_rt_t = [269.4017, 275.7074, 266.4612, 266.0675, 259.3810, 254.6624, 259.3924, 254.1609, 263.1543];
std_rt_t = [23.8863, 19.9177, 21.9598, 19.9657, 20.6684, 22.4727, 22.6608, 21.4406, 21.6215];

% Perform ANOVA
[p, tbl, stats] = anova1([mean_rt_t', std_rt_t']);

% Display the ANOVA table
disp('ANOVA Table:')
disp(tbl)

% Interpretation of p-value
if p < 0.05
    disp('The p-value is less than 0.05, indicating that there is a statistically significant difference among the means of the groups.')
else
    disp('The p-value is greater than or equal to 0.05, indicating that there is no statistically significant difference among the means of the groups.')
end

%% step 8: plot eye posistion on hand posistion plot
%extract relevant trial data for eye movement
hep=cell([1,length(R)]);
vep=cell([1,length(R)]);
mean_hep=zeros([1,length(R)]);
mean_vep=zeros([1,length(R)]);
for n=1:length(R)
    hep{n}=R(n).hep(timegocue(n):timetargetacquire(n));
    vep{n}=R(n).vep(timegocue(n):timetargetacquire(n));

    %calculate mean
    mean_hep(n)=mean(hep{n});
    mean_vep(n)=mean(vep{n});
end

hold on
for n=1:length(R)
%target 1
if allTarget1X(n)==Target1X(1)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor','r')
end

%target 2
if allTarget1X(n)==Target1X(2)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.8500 0.3250 0.0980])
end

%target 3
if allTarget1X(n)==Target1X(3)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.9290 0.6940 0.1250])
end

%target 4
if allTarget1X(n)==Target1X(4)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.4940 0.1840 0.5560])
end

%target 5
if allTarget1X(n)==Target1X(5)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.4660 0.6740 0.1880])
end

%target 6
if allTarget1X(n)==Target1X(6)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.3010 0.7450 0.9330])
end

%target 7
if allTarget1X(n)==Target1X(7)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[0.6350 0.0780 0.1840])
end

%target 8
if allTarget1X(n)==Target1X(8)
plot(mean_hep(n),mean_vep(n),'o','MarkerEdgeColor',[0,0,1],'MarkerFaceColor',[1 1 1])
end
end



%% Part 2, neural data analysis Step 8
figure;
hold on 
colors2=gray(20)
%pThere are four cells present, plot raster plots for a sample trial (1000)
plot([[R(1000).cells(1).spikeTimes] [R(1000).cells(1).spikeTimes]],[-20,240],'color',colors2(10,:))
% text(min(R(1000).cells(1).spikeTimes), 0, 'Cell 1', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
plot([[R(1000).cells(2).spikeTimes] [R(1000).cells(2).spikeTimes]],[120,260],'color',colors2(10,:))
% text(min(R(1000).cells(2).spikeTimes), 190, 'Cell 2', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
plot([[R(1000).cells(3).spikeTimes] [R(1000).cells(3).spikeTimes]],[260,400],'color',colors2(10,:))
% text(min(R(1000).cells(3).spikeTimes), 330, 'Cell 3', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
plot([[R(1000).cells(4).spikeTimes] [R(1000).cells(4).spikeTimes]],[400,540],'color',colors2(10,:))
% text(min(R(1000).cells(4).spikeTimes), 470, 'Cell 4', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');


%plot cues
l1=xline(R(1000).timeCueOnset,'Color',[0.6350 0.0780 0.1840],'LineWidth',2);
l2=xline(R(1000).timeGoCuePHOTO,'Color',[0 0.4470 0.7410],'LineWidth',2);

%plot hand and eye movement above rasters

p1=plot([R(1000).hhp]+4*140,'r','LineWidth',1.5);
p2=plot([R(1000).vhp]+4*140,'b','LineWidth',1.5);
p3=plot(movmean([R(1000).hep],51)+4*140,'m','LineWidth',1.5);
p4=plot(movmean([R(1000).vep],51)+4*140,'c','LineWidth',1.5);

legend([p1(1) p2(1) p3(1) p4(1) l1(1) l2(1)],'Horizontal Hand Position','Vertical Hand Position','Horizontal Eye Position','Vertical Eye Position','timeCueOnset','timeGoCuePHOTO','Location','eastoutside')

xlim([0,length([R(1000).hhp])])
ylim([-20,680])

xlabel('Time (ms)')
title('Raster Plot (#1000)', 'FontSize',25,'Color','b')
yticks([50,190,330,470,610])
yticklabels({'Cell 1 (Spike Count)','Cell 2 (Spike Count)','Cell 3 (Spike Count)','Cell 4 (Spike Count)','Position Data in mm'})

%%
%define cue onset
timeonset=[R.timeCueOnset];
%extract trials for upper right target
upperright=find(allTarget1X==Target1X(7));

%initialize psth matrix
psth=zeros([4,length(upperright),900]);

%extract data from each trial in 900ms window for each cell
spikeTimes=cell([1,4]);
w_align=zeros(1,length(upperright));
for n=1:4
for n=1:length(upperright)
    %find 900ms window of data
    id1=find(R(upperright(n)).cells(n).spikeTimes==interp1(R(upperright(n)).cells(n).spikeTimes,R(upperright(n)).cells(n).spikeTimes,timeonset(upperright(n))-300,'next'));
    id2=find(R(upperright(n)).cells(n).spikeTimes==interp1(R(upperright(n)).cells(n).spikeTimes,R(upperright(n)).cells(n).spikeTimes,timeonset(upperright(n))+600,'previous'));
    %extract data
    spikeTimes{n}{n}=R(upperright(n)).cells(n).spikeTimes(id1:id2);
    %window alignment for each trial
    w_align(n)=timeonset(upperright(n))-301;
    %align data
    spikeTimes{n}{n}=spikeTimes{n}{n}-w_align(n);
    %round data
    spikeTimes{n}{n}=round(spikeTimes{n}{n});
    %truncate at 1 and 900
    spikeTimes{n}{n}(spikeTimes{n}{n}<1)=1;
    spikeTimes{n}{n}(spikeTimes{n}{n}>900)=900;
    %populate matrix
    for j=1:length(spikeTimes{n}{n})
        psth(n,n,spikeTimes{n}{n}(j))=psth(n,n,spikeTimes{n}{n}(j))+1;
    end
    %apply gaussian kernel
    w=gausswin(10, 2.5);
    w=w/sum(w);
    psth(n,n,:)=filter(w,1,psth(n,n,:));
end
end

%%
%sum psth across trials
psth=squeeze(sum(psth,2));

%%
%plot results
for n=1:4
    subplot(2,2,n)
    bar(psth(n,:))
    title(append('Cell ',num2str(n)))
    xlim([300,400])
end


xlabel('Time (ms)')
ylabel('Sum Spike Count Over All Trials')
title('PSTH')

%%
%convert target locations to degrees
xt=[-64,-98,-86,-34,34,86,98,64];
yt=[Target1Y,flip(Target1Y)];
%extract target degrees
degrees=zeros([1,8]);
for n=1:8
    id=find(allTarget1X==xt(n));
    degrees(n)=R(id(1)).TrialParams.targetAngularDirection;
end

%sum spikes during 500ms window over all trials
spikeCount=zeros([4,length(R)]);
for n=1:4
for n=1:length(R)
    %find 500ms window of data
    id1=find(R(n).cells(n).spikeTimes==interp1(R(n).cells(n).spikeTimes,R(n).cells(n).spikeTimes,timeonset(n)+100,'next'));
    id2=find(R(n).cells(n).spikeTimes==interp1(R(n).cells(n).spikeTimes,R(n).cells(n).spikeTimes,timeonset(n)+600,'previous'));
    %calculate total spikes
    if isempty(id2-id1)
        spikeCount(n,n)=0;
    else
        spikeCount(n,n)=id2-id1;
    end
end
end

%calculate tuning curve points
tuningcurve=zeros([1,8]);
for n=1:4
for n=1:8
    tuningcurve(n,n)=mean(spikeCount(n,allTarget1X==xt(n)));
end
end
%%
%
%sum spikes during 500ms window over all trials
spikeCount=zeros([4,length(R)]);
for n=1:4
for i=1:length(R)
%fit cosine curve
x=[1,8];
y=tuningcurve(n,:);
yu=max(y);
yl=min(y);
yr=(yu-yl);                               
yz=y-yu+(yr/2);
zx=x(yz(:).*circshift(yz(:),[1 0])<= 0);     
per=2*mean(diff(zx));                     
ym=mean(y);                             
fit= @(b,x)  b(1).*(sin(2*pi*x./b(2) + 2*pi/b(3))) + b(4);    
fcn= @(b) sum((fit(b,x) - y).^2);                              
s=fminsearch(fcn, [yr;  per;  -1;  ym]) ;                     
xp=linspace(min(x),max(x));

%plot
plot(x,tuningcurve(n,:),'x',xp,fit(s,xp),'r')
title(append('Cell ',num2str(n)))
end
end

xlabel('Degrees')
ylabel('Spikes per second')
title('Tuning Curves')


%% tuning curve: https://www.mathworks.com/matlabcentral/answers/121579-curve-fitting-to-a-sinusoidal-function
%%
% Preallocate array for tuning curve points
tuning_curve_points = zeros(4, 8);

% Calculate spike counts for each cell and target location
for cell_idx = 1:4
    for target_idx = 1:8
        % Find trials corresponding to the current target location
        target_trials = find(allTarget1X == xt(target_idx));

        % Calculate mean spike count for the current cell and target location
        tuning_curve_points(cell_idx, target_idx) = mean(spikeCount(cell_idx, target_trials));
    end
end

% Plot tuning curves
figure;
for cell_idx = 1:4
    subplot(4, 1, cell_idx);
    
    % Fit cosine curve to tuning curve points
    x = [1:8];
    y = tuning_curve_points(cell_idx, :);
    y_max = max(y);
    y_min = min(y);
    y_range = y_max - y_min;
    y_zeroed = y - y_max + (y_range / 2);
    zero_crossings = x(y_zeroed(:) .* circshift(y_zeroed(:), [1, 0]) <= 0);
    period = 2 * mean(diff(zero_crossings));
    y_mean = mean(y);
    fit_function = @(b, x) b(1) * (sin(2 * pi * x ./ b(2) + 2 * pi / b(3))) + b(4);
    fitting_cost_function = @(b) sum((fit_function(b, x) - y).^2);
    initial_guess = [y_range; period; -1; y_mean];
    optimal_parameters = fminsearch(fitting_cost_function, initial_guess);
    x_values = linspace(min(x), max(x));

    % Plot tuning curve points and fitted curve
    plot(x, y, 'x', x_values, fit_function(optimal_parameters, x_values), 'r');
    title(['Cell ', num2str(cell_idx)]);
    xlabel('Degrees');
    ylabel('Spikes per second');
end

sgtitle('Tuning Curves for all Cells');

%%
% Define cue onset time
timeonset = [R.timeCueOnset];

% Extract trials for upper right target
ts = find(allTarget1X == Target1X(7));

% Initialize spike count matrix
spikeCount = zeros(4, length(R));

% Calculate spike count for each cell and trial within a 500ms window
for k = 1:4
    for i = 1:length(R)
        % Find spike times within 500ms window
        id1 = find(R(i).cells(k).spikeTimes == interp1(R(i).cells(k).spikeTimes, R(i).cells(k).spikeTimes, timeCO(i) + 100, 'next'));
        id2 = find(R(i).cells(k).spikeTimes == interp1(R(i).cells(k).spikeTimes, R(i).cells(k).spikeTimes, timeCO(i) + 600, 'previous'));

        % Calculate total spikes
        spikeCount(k, i) = numel(id1:id2);
    end
end

% Calculate mean spike count across trials for each cell and target location
meanSpikeCount = zeros(4, length(Target1X));
for k = 1:4
    for i = 1:length(uniTarget1X)
        meanSpikeCount(k, i) = mean(spikeCount(k, allTarget1X == Target1X(i)));
    end
end

% Convert target locations to degrees
xt = [-64, -98, -86, -34, 34, 86, 98, 64];
yt = [uniTarget1Y, flip(Target1Y)];
degrees = atan2d(yt, xt);

% Plot tuning curves
fig6 = figure(6);
for k = 1:4
    subplot(2, 2, k)

    % Fit cosine curve
    x = degrees;
    y = meanSpikeCount(k, :);
    yu = max(y);
    yl = min(y);
    yr = (yu - yl);
    yz = y - yu + (yr / 2);
    zx = x(yz(:) .* circshift(yz(:), [1, 0]) <= 0);
    per = 2 * mean(diff(zx));
    ym = mean(y);
    fit = @(b, x)  b(1) .* (sin(2 * pi * x ./ b(2) + 2 * pi / b(3))) + b(4);
    fcn = @(b) sum((fit(b, x) - y).^2);
    s = fminsearch(fcn, [yr; per; -1; ym]);
    xp = linspace(min(x), max(x));

    % Plot
    plot(x, y, 'x', xp, fit(s, xp), 'r')
    title(['Cell ', num2str(k)])
end

% Add common labels and title
h = axes(fig6, 'visible', 'off');
h.XLabel.Visible = 'on';
h.YLabel.Visible = 'on';
h.Title.Visible = 'on';
xlabel(h, 'Degrees')
ylabel(h, 'Spikes per second')
title('Tuning Curves')

%%
% Preallocate array for tuning curve points
tuning_curve_points = zeros(4, 8);

% Calculate spike counts for each cell and target location
for cell_idx = 1:4
    for target_idx = 1:8
        % Find trials corresponding to the current target location
        target_trials = find(allTarget1X == xt(target_idx));

        % Calculate mean spike count for the current cell and target location
        tuning_curve_points(cell_idx, target_idx) = mean(spikeCount(cell_idx, target_trials));
    end
end

% Plot tuning curves
figure;
for cell_idx = 1:4
    subplot(4, 1, cell_idx);

    % Fit cosine curve to tuning curve points
    x = [1:8];
    y = tuning_curve_points(cell_idx, :);
    y_max = max(y);
    y_min = min(y);
    y_range = y_max - y_min;
    y_zeroed = y - y_max + (y_range / 2);
    zero_crossings = x(y_zeroed(:) .* circshift(y_zeroed(:), [1, 0]) <= 0);
    period = 2 * mean(diff(zero_crossings));
    y_mean = mean(y);

    % Define fitting function
    fit_function = @(b, x) b(1) * (sin(2 * pi * x ./ b(2) + 2 * pi / b(3))) + b(4);

    % Define cost function for fitting
    fitting_cost_function = @(b) sum((fit_function(b, x) - y).^2);

    % Initial parameter guesses
    initial_guess = [y_range; period; -1; y_mean];

    % Lower and upper bounds for parameters
    lb = [0; period - 1; -Inf; -Inf];
    ub = [2 * y_range; period + 1; Inf; Inf];

    % Perform constrained optimization
    options = optimset('Display', 'off');
    optimal_parameters = fmincon(fitting_cost_function, initial_guess, [], [], [], [], lb, ub, [], options);

    % Generate x values for smooth curve
    x_values = linspace(min(x), max(x));

    % Plot tuning curve points and fitted curve
    plot(x, y, 'x', x_values, fit_function(optimal_parameters, x_values), 'r');
    title(['Cell ', num2str(cell_idx)]);
    xlabel('Degrees');
    ylabel('Spikes per second');
end

sgtitle('Tuning Curves for all Cells');

