%% Code for generating RF image of a object in an experimental setup
% This is always the most updated version
% C is modified a bit
function [] = main_od
clear all;

%% Defining all variables here
c = physconst('LightSpeed');
f = 'f3';
Ntag = 16; 
Nrecv = 4;
object = 'circlefill';
SNR=20; %unit dB 
l = 1;% Defining l; Scaling factor for the image area
b = 1;% Defining scaling for receivers
m = 1;% Choosing scaling factor of scatterer position
k1 = 0.15; % Choosing shifting factor of scatterer position in x and y direction
k2 = -0.22;
th=0.6;
RCS=.5;
%% Starting code here, position of tags, receivers and object scatterers

switch(f)
    case 'f1'
        % Working Frequency range f1 = 900MHz, Second harmonic f2 = 1.8GHz,
        % 21 frequencies
        Freq_v = (0.2:0.05:2.4)*1e9;
        
    case 'f2'
        % Working Frequency range f1 = 700MHz, Second harmonic f2 = 1.4GHz
        % 51 frequencies
        Freq_v = [ 2.0 2.1 2.2 2.3 2.4]*1e9;
        
    case 'f3'
        % Working Frequency range f1 = 900MHz, Second harmonic f2 = 1.8GHz,
        % 11 frequencies
        Freq_v =(1.7:0.05:2.2)*1e9;
        
    case 'f4'
        % Freq = 2*[721 750 829 889 950 996 1045 1060] MHz BW=38% 
        Freq_v = 2*[0.721 0.750 0.829 0.889 0.950 0.996 1.045 1.060]*1e9;
        
    case 'f5'
        Freq_v = [1.79 2.03 2.2]*1e9; % 0.9 1.5 2.2
        
    case 'f6'
        Freq_v= 2e9;
        
    otherwise
        display('Error in frequency');
end
Nfreq = length(Freq_v);
assignin('base','Freq_v',Freq_v);
% Tags are arranged uniformly spaced on a l x l square

% Allocating size of Pos_tag 
Pos_tag = zeros(Ntag,2);

switch(Ntag)
    case 32
        % Try to improve resolution by shifting tags by less than
        % wavelength (shifting by +/-3cm)
for i=1:Ntag/4
    Pos_tag(i, :)=[-0.35+(i-1)*0.1 -0.5] ;
end
for i=1:Ntag/4
    Pos_tag(i+(Ntag/4),:)=[0.5 -0.35+(i-1)*0.1] ;
end
for i=1:Ntag/4
    Pos_tag(i+(2*Ntag/4), :)=[-0.35+(i-1)*0.1 0.5] ;
end
for i=1:Ntag/4
    Pos_tag(i+(3*Ntag/4), :)=[-0.5 -0.35+(i-1)*0.1] ;
end

% Take care. Symmetricity of tags and receiver is serious for less
% distortion
    case 16     
for i=1:Ntag/4
    Pos_tag(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
end
for i=1:Ntag/4
    Pos_tag(i+(Ntag/4),:)=[0.5 -0.3+(i-1)*0.2] ;
end
for i=1:Ntag/4
    Pos_tag(i+(2*Ntag/4), :)=[-0.3+(i-1)*0.2 0.5] ;
end
for i=1:Ntag/4
    Pos_tag(i+(3*Ntag/4), :)=[-0.5 -0.3+(i-1)*0.2] ;
end

    case 4
        Pos_tag = [-0.5,0.5;-0.1667,0.5;0.1667,0.5;0.5,0.5];
   
    otherwise
        disp('Error in Pos_tag');
end
Pos_tag = l* Pos_tag;
% What is Pos_tag_mean. Gives same result as Pos_tag
Pos_tag_mean=Pos_tag+2*(rand(Ntag,2)-1)*0.00; 
Pos_recv = zeros(Nrecv,2);
switch(Nrecv)
    case 1
        Pos_recv = [-0.5,-0.5];
    case 2
        Pos_recv = [-0.5,-0.5;0.5,-0.5];   
    case 4
        % Receivers are arranged on the four corners of 1 x 1 square
        Pos_recv = [-0.5,-0.5;0.5,-0.5;0.5,0.5;-0.5,0.5];
    case 12
        Pos_recv = zeros(Nrecv,2);
        for i=1:Nrecv/3
            Pos_recv(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
        end
        for i=1:Nrecv/3
            Pos_recv(i+(Nrecv/3),:)=[0.5 -0.3+(i-1)*0.2] ;
        end
        for i=1:Nrecv/3
            Pos_recv(i+(2*Nrecv/3), :)=[-0.5 -0.3+(i-1)*0.2] ;
        end   
    case 16
        for i=1:Nrecv/4
            Pos_recv(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(Nrecv/4),:)=[0.5 -0.3+(i-1)*0.2] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(2*Nrecv/4), :)=[-0.3+(i-1)*0.2 0.5] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(3*Nrecv/4), :)=[-0.5 -0.3+(i-1)*0.2] ;
        end
    case 32
        for i=1:Nrecv/4
            Pos_recv(i, :)=[-0.35+(i-1)*0.1 -0.5] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(Nrecv/4),:)=[0.5 -0.35+(i-1)*0.1] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(2*Nrecv/4), :)=[-0.35+(i-1)*0.1 0.5] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(3*Nrecv/4), :)=[-0.5 -0.35+(i-1)*0.1] ;
        end

        
    otherwise
        disp('Error in Pos_recv');
end

Pos_recv = b* Pos_recv;
 
% Plotting tags and receiver
figure;
plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
hold on;
plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);

switch(object)
    case 'var'
        
        Pos_s = [0.0, 0; 
                0.02, 0.25;
                -0.1, 0.06;
                0.18,0.12;
                -0.28,-0.25;
                0.3,-0.1;
                -0.08,-0.3;
                ]; 
 
    case 'T4'
        
        % Defining number of scattereres and corresponding positions
        % Ns = 4;
        Pos_s=[-0.105000000000000,0;
               0.0750000000000000,0;
               0.200000000000000,-0.105000000000000;
               0.200000000000000,0.0750000000000000];
    case 'L12'
        
        % Defining number of scatterers and corresponding positions
        % Ns = 12;
        Pos_s=[-0.105,0;
               -0.075,0;
               -0.045,0;
               -0.015,0;
               0.0150,0;
               0.0450,0;
               0.0750,0;
               0.1050,0;
               0.1350,0;
               0.1650,0;
               0.1950,0;
               0.195,-0.03;
               0.195,-0.06;
               0.195,-0.09;
               0.195,-0.12];
    case 'L4'
        
        % Defining number of scatterers and corresponding positions
        % Ns = 4
        Pos_s = [-0.105,0;
                 0.075,0;
                 0.2,0.015;
                 0.2,0.105]; 
        
    case 'box1'
        
        % Ns = 4
        Pos_s = [-0.05,-0.05;
                 0.05,-0.05;
                 0.05,0.05;
                 -0.05,0.05];
   
        
    case 'circlefill'
        
        Pos_s=[];
        i=0;
        dtheta=1.2*pi/8;
        
        a0 = 0.05; % radius = a0*m
        % The radius of the object is a*m = 20*.25 cm = 5 cm
        for a = 0.01:0.02:a0
            theta=0;
            while theta<=2*pi-dtheta/2
                i=i+1;
                % Parametric form is (X,Y)=a(cos t, sin t)
                Pos_s(i,:)=(a.*[cos(theta) sin(theta)]);
                theta=theta+dtheta;
            end
            i=i+1;
            dtheta=dtheta-pi/64;
        end
%         Pos_s=sd_round(Pos_s); 
        
        %Radius
        ro=a*m*100;
        assignin('base','ro',ro);
        
    case 'circleouter'
         Pos_s=[];
        i=0;
        dtheta=pi/8;
        % The radius of the object is a*m = 20*.25 cm = 5 cm
        a=0.2;
        theta=0;
        while theta<=2*pi-dtheta/2
            i=i+1;
            % Parametric form is (X,Y)=a(cos t, sin t)
            Pos_s(i,:)=(a.*[cos(theta) sin(theta)]);
            theta=theta+dtheta;
        end
                
        %Radius
        ro=a*m*100;
        assignin('base','ro',ro);
        
    case 'ellipsefill'
        Pos_s=[];
        i=0;
        dtheta=pi/8;
        
        % The semi-major and semi-minor axes are (a, b)*m
        % phi is the angle by which it is rotated
        % Parametric form is (X,Y)=(a cos t, b sin t)
        % Parametric form for rotated is (X,Y)= (a*cos t*cos phi-b*sin t
        % sin phi, a*cos t*sin phi+b*sin t*cos phi)
        
        phi=pi/3;
        b=0.018;
        
        for a = 0.01:0.02:0.06
            theta=0;
            while theta<=2*pi-dtheta/2
                i=i+1;
                % Parametric form is (X,Y)=a(cos t, sin t)
                Pos_s(i,:)=[a.*cos(theta).*cos(phi)-b.*sin(theta).*sin(phi), a.*cos(theta).*sin(phi)+b.*sin(theta).*cos(phi)];
                theta=theta+dtheta;
            end
            i=i+1;
%             dtheta=dtheta-pi/32;
        end
        ao=a*m*100;
        bo=b*m*100;
        assignin('base','ao',ao);
        assignin('base','bo',bo);
        
    case 'boxfill'
        Pos_s=[];
        i=1;
        for x=-0.2:0.05:0.2
            for y=-0.2:0.05:0.2
                Pos_s(i,:)=[x y];
                i=i+1;
            end
            i=i+1;
        end
                 
    otherwise 
        disp('Error in Pos_s');
   
end

% Scaling scatterer position by m
Pos_s = m * Pos_s;
[m1,~]=size(Pos_s);
% Shifting Pos_s by k
Pos_s = Pos_s - k1* [zeros(m1,1), ones(m1,1)];
Pos_s = Pos_s - k2* [ones(m1,1), zeros(m1,1)];
% Plotting object
plot(Pos_s(:,1),Pos_s(:,2),'k*');

axis 'square'
axis 'tight'
legend('Pos tag','Pos recv','Pos Scatterer');

%% 
% Compute phi_tag and phi_recv
% phi_tag is angle in radians from each tag to mean scatterer position
% phi_recv is angle in radians from each receiver to mean scatterer position

x_sc=mean(Pos_s(:,1));
y_sc=mean(Pos_s(:,2));
phi_tag=zeros(1,Ntag);

% Measuring time for this section
tic;
for i=1:Ntag
    x_d=Pos_tag(i,1)-x_sc;
    y_d=Pos_tag(i,2)-y_sc;
    phi_tag(1,i)=angle(x_d+sqrt(-1)*y_d);
end
phi_recv=zeros(1,Nrecv);
for i=1:Nrecv
    x_d=Pos_recv(i,1)-x_sc;
    y_d=Pos_recv(i,2)-y_sc;
    phi_recv(1,i)= angle(x_d+sqrt(-1)*y_d);
end

t1 = toc;


disp(['This block computed phi_tag and phi_recv in',num2str(t1),'secs']);

%% This is K = (Kx,Ky). Refer to paper

% Kx = f/c * [cos(phi_tag) + cos(phi_recv)]
Kx=zeros(Ntag*Nrecv*Nfreq);

% Ky = f/c * [sin(phi_tag) + sin(phi_recv)]
Ky=zeros(Ntag*Nrecv*Nfreq);

count = 1;
for i=1:length(Freq_v)
    for m=1:length(phi_tag)
        for n=1:length(phi_recv)
            Kx(count) = Freq_v(i)/c*(cos(phi_tag(m))+cos(phi_recv(n)));
            Ky(count) = Freq_v(i)/c*(sin(phi_tag(m))+sin(phi_recv(n)));
            count = count+1;
        end
    end
end

figure;
plot(Kx,Ky,'r+');
hold on;
theta = 0:0.01:2*pi;
x = 2*max(Freq_v)/c*cos(theta);
y = 2*max(Freq_v)/c*sin(theta);

plot(x, y)
axis square

t1 = toc;

disp(['This block computed and plotted K-space in ',num2str(t1),' secs.']);

%% generate rician fading factors
% Rician fading is CN(1, variance)
% variance of real/ imaginary is half of total, hence factor of 1/2 * 
% SNR (actual converted from dB)
An=sqrt(1/2*10^(-SNR/10));
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
Rician_Factor=zeros(k1,k2,length(Freq_v));
Rician_Factor_Phase=zeros(length(Freq_v)*k1*k2,1);
Rician_Factor_Amp=zeros(length(Freq_v)*k1*k2,1);
count =1;
tic;

for i=1:length(Freq_v)
    for m=1:k1
        for n=1:k2
            X=1+An*randn(1,1);  % is 1 for LOS? (Rician Fading)
            Y=An*randn(1,1);
            Z=(X+sqrt(-1)*Y);
            Rician_Factor_Phase(count)= angle(Z);
            Rician_Factor_Amp(count)= abs(Z);
            Rician_Factor(m,n,i)=Z;
            count=count+1;
        end
    end
end
    
t1 = toc;


disp(['This block computed Rician Factor in ',num2str(t1),' secs.']);


%% generate signal for w/o and w/ scatterers case

G_wo_s=generation_signal_wo_scatter(Freq_v,Pos_tag,Pos_recv,Rician_Factor);

G_w_s=generation_signal_w_scatter(Freq_v,Pos_tag,Pos_recv,Pos_s,RCS,Rician_Factor);

%% generate signal after calibration 
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
G_calib=zeros(k1,k2,length(Freq_v));

tic;

for i=1:length(Freq_v)
    freq=Freq_v(i);
    lamda=c/freq;
    for n=1:k2
        for m=1:k1
            p_tag=Pos_tag(m,:);
            p_recv=Pos_recv(n,:);
            r=norm(p_tag-p_recv);
            for n_pair=1:k2
                if n_pair~=n
                    G_w_s_ratio=G_w_s(m,n,i)/G_w_s(m,n_pair,i);
                    G_wo_s_ratio=G_wo_s(m,n,i)/G_wo_s(m,n_pair,i);
                    G_calib(m,n,i)=G_calib(m,n,i)+(G_w_s_ratio/G_wo_s_ratio-1)*exp(-1j*2*pi/lamda*r);
                end
            end
        end
    end
end

t1 = toc;
assignin('base','G_calib',G_calib);

disp(['This block Signal After Calibration in ',num2str(t1),' secs.']);

%% Image recovery

% x_v=(-.5:0.005: .5);
x_v = linspace(-0.5,0.5,100);
y_v=x_v;
lx=length(x_v);

image_output=zeros(lx^2,1);

disp(' Beginning of Image recovery code block. ');

tic;

% improved version from here

minfreq=min(Freq_v);
ratio_freq2=(Freq_v./minfreq).^2;
ratio_phi=zeros(k1,k2);
const=zeros(k1,k2,length(Freq_v));
exp_const=Freq_v.*(1j*2*pi/c);

% This P finds all possible combinations of x_v,y_v and arranges as 
% [all_x first_y; all_x first_y;..]

p=combvec(x_v,y_v)';
assignin('base','p',p);
% Read MatlabTip
% R1 is point to tag Euclidean distance
R1 = sqrt(sum(abs( repmat(permute(p, [1 3 2]), [1 Ntag 1]) ...
- repmat(permute(Pos_tag_mean, [3 1 2]), [lx*lx 1 1]) ).^2, 3));

% R2 is point to receiver Euclidean distance
R2 = sqrt(sum(abs( repmat(permute(p, [1 3 2]), [1 Nrecv 1]) ...
- repmat(permute(Pos_recv, [3 1 2]), [lx*lx 1 1]) ).^2, 3));
assignin('base','R1',R1)
assignin('base','R2',R2)

for n=1:k2
    for m=1:k1
         ratio_phi(m,n)=1;
         for i=1:length(Freq_v)
           const(m,n,i) = ratio_freq2(i)*ratio_phi(m,n)*G_calib(m,n,i);
           image_output = image_output+const(m,n,i).*exp(exp_const(i) ...
               .*(R1(:,m)+R2(:,n)));
        end
    end
end


image_output=reshape(image_output,[lx,lx]);
assignin('base','image_output',image_output);

t1=toc;
disp(['Total time taken is',num2str(t1),'secs.']);
%% Getting Object

% mesh(x_v,y_v,(abs(image_output).^2)');
% view(2);

C= (abs(image_output).^2)';
cmax = max(C(:)); %Making range 0-1, easier for thresholding
cmin = min(C(:));

C =(C-cmin)/(cmax-cmin); 
% This sends my variable to the workspace
assignin('base','image_output',image_output);
assignin('base','C',C);

% Plotting figure here
figure;
mesh(C); % for a better view, the LSF is displayed upside down
axis 'square'; axis 'tight'
hold all
% Defining contour with threshold and P is contour matrix
P = contour(C, [th th],'color',[0 0.5 0],'LineWidth',1);
assignin('base','P',P);

end

%% generating signal without scatterer

function [G] = generation_signal_wo_scatter(Freq_v,Pos_tag,Pos_recv,Rician_Factor)

c=3e8;
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
G=zeros(k1,k2,length(Freq_v));
for i=1:length(Freq_v)
    for m=1:k1
        for n=1:k2
            freq=Freq_v(i);
            lamda=c/freq;
            p_tag=Pos_tag(m,:);
            p_recv=Pos_recv(n,:);
            r=norm(p_tag-p_recv);
            
            G(m,n,i)=Rician_Factor(m,n,i)*lamda/(4*pi*r)*exp(-1j*2*pi/lamda*r);
            if r == 0
                G(m,n,i) = 1e-4+1j*1e-4;
                
            end
        end
    end
end
end
%% generating signal with scatterer 

function [G] = generation_signal_w_scatter(Freq_v,Pos_tag,Pos_recv,Pos_s,RCS,Rician_Factor)
c=3e8;
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
[k3, ~]=size(Pos_s);
G=zeros(k1,k2,length(Freq_v));
for i=1:length(Freq_v)
    for m=1:k1
        for n=1:k2
            freq=Freq_v(i);
            lamda=c/freq;
            p_tag=Pos_tag(m,:);
            p_recv=Pos_recv(n,:);
            r=norm(p_tag-p_recv);
%             G(m,n,i) = 0;
            G(m,n,i)=Rician_Factor(m,n,i)*lamda/(4*pi*r)*exp(-1j*2*pi/lamda*r);
            if r ==0
                G(m,n,i) = 1e-4+1j*1e-4;
            end
            for u=1:k3
                p_s=Pos_s(u,:);
                r1=norm(p_tag-p_s);
                r2=norm(p_s-p_recv);
                G(m,n,i)=G(m,n,i)+sqrt(RCS/4/pi/r1^2)*lamda/(4*pi*r2)*exp(-1j*2*pi/lamda*(r1+r2));
            end
        end
    end
end
end






