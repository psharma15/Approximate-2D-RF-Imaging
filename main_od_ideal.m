% Code for generating RF image of a object in an experimental setup
% This is hypothetical model, only signals coming from target are used to
% reconstruct and no fading/ power loss with distance is assumed
% Pragya Sharma, Cornell University, Jan. 26, 2017
function [] = main_od_ideal
clear all;

%% Generating forward problem
% Defining all variables here
c = physconst('LightSpeed');
f = 'f6';
Ntag = 16; 
Nrecv = 2;
object = 'circlefill';
l = 1; % Defining l; Scaling factor for the image area
b = 1; % Defining scaling for receivers
m = 1; % Choosing scaling factor of scatterer position
k1 = 0.1; % Choosing shifting factor of scatterer pos in x and y dir
k2 = -0.05;
RCS=0.1;

% position of tags, receivers and object scatterers
switch(f)
    case 'f3'
        Freq =(1.7:0.05:2.2)*1e9;
    case 'f5'
        Freq = [1.79 2.03 2.2]*1e9; 
    case 'f6'
        Freq = 2e9;
    otherwise
        display('Error in frequency');
end
Nfreq = length(Freq);
assignin('base','Freq_v',Freq);
Pos_tag = zeros(Ntag,2);
switch(Ntag)
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
switch(Nrecv)
    case 1
        Pos_recv = [-0.5,-0.5];
    case 2
        Pos_recv = [-0.5,-0.5;0.5,-0.5];   
    case 4
        Pos_recv = [-0.5,-0.5;0.5,-0.5;0.5,0.5;-0.5,0.5];
    otherwise
        disp('Error in Pos_recv');
end
Pos_recv = b* Pos_recv;
figure;
plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
hold on;
plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);
switch(object)
    case 'var'
        Pos_s = [0.0,0; 0.025,0;0,0.025;0.025,0.025]; 
    case 'circlefill'
        Pos_s=[];
        i=0;
        dtheta=1.2*pi/8;
        a0 = 0.05; % radius = a0*m
        for a = 0.01:0.02:a0
            theta=0;
            while theta<=2*pi-dtheta/2
                i=i+1;
                Pos_s(i,:)=(a.*[cos(theta) sin(theta)]);
                theta=theta+dtheta;
            end
            i=i+1;
            dtheta=dtheta-pi/64;
        end
        ro=a*m*100;
        assignin('base','ro',ro);
    case 'ellipsefill'
        Pos_s=[];
        i=0;
        dtheta=pi/8;
        phi=91/4;
        b=0.06;
        for a = 0.05:0.05:0.25
            theta=0;
            while theta<=2*pi-dtheta/2
                i=i+1;
                Pos_s(i,:)=[a.*cos(theta).*cos(phi)-b.*sin(theta).*sin(phi), a.*cos(theta).*sin(phi)+b.*sin(theta).*cos(phi)];
                theta=theta+dtheta;
            end
            i=i+1;
        end
        ao=a*m*100;
        bo=b*m*100;
        assignin('base','ao',ao);
        assignin('base','bo',bo);
    otherwise 
        disp('Error in Pos_s');
end
Pos_s = m * Pos_s;
[m1,~]=size(Pos_s);
Pos_s = Pos_s - k1* [zeros(m1,1), ones(m1,1)];
Pos_s = Pos_s - k2* [ones(m1,1), zeros(m1,1)];
plot(Pos_s(:,1),Pos_s(:,2),'k*');
axis 'square'
axis 'tight'
legend('Pos tag','Pos recv','Pos Scatterer');

% Compute phi_tag and phi_recv
x_sc=mean(Pos_s(:,1));
y_sc=mean(Pos_s(:,2));
phi_tag=zeros(1,Ntag);
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

% This is K = (Kx,Ky). 
Kx=zeros(Ntag*Nrecv*Nfreq);
Ky=zeros(Ntag*Nrecv*Nfreq);
count = 1;
for i=1:length(Freq)
    for m=1:length(phi_tag)
        for n=1:length(phi_recv)
            Kx(count) = Freq(i)/c*(cos(phi_tag(m))+cos(phi_recv(n)));
            Ky(count) = Freq(i)/c*(sin(phi_tag(m))+sin(phi_recv(n)));
            count = count+1;
        end
    end
end
figure;
plot(Kx,Ky,'r+');
hold on;
theta = 0:0.01:2*pi;
x = 2*max(Freq)/c*cos(theta);
y = 2*max(Freq)/c*sin(theta);
plot(x, y)
axis square

% generate signal for w/o and w/ scatterers case
G_w_s=generation_signal_w_scatter(Freq,Pos_tag,Pos_recv,Pos_s,RCS);


%% Recovery Section
G = reshape(G_w_s,[],1); % No G_calib with 1 Recv!
assignin('base','G',G)
V = G;
x_v = linspace(-0.15,0.15,32);
y_v = x_v;
lx = length(x_v);
P = combvec(x_v,y_v)';
R1 = repmat(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Ntag 1]) ...
- repmat(permute(Pos_tag, [3 1 2]), [lx*lx 1 1]) ).^2, 3)),1,Nrecv);
dummy1 = ones(1,Ntag);
R2 = kron(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Nrecv 1]) ...
- repmat(permute(Pos_recv, [3 1 2]), [lx*lx 1 1]) ).^2, 3)), dummy1);
assignin('base','R1',R1);
assignin('base','R2',R2);
R = R1 + R2;
R = repmat(R,1,Nfreq);
dummy3 = ones(1,Ntag*Nrecv);
Freq = repmat(kron(Freq,dummy3),[lx*lx 1]);
e = exp(-1j*(2*pi/c)*(Freq.*R));
e = transpose(e);
assignin('base','e',e);
B = diag(e'*(V*V')*e);
assignin('base','B',B);
B = real(B);
B = (reshape(B, [lx lx]))';
[X, Y] = meshgrid(x_v,y_v);
figure;
mesh(X,Y,B)
axis 'tight'
axis 'square'
end

% generating signal with scatterer 
function [G] = generation_signal_w_scatter(Freq_v,Pos_tag,Pos_recv,Pos_s,RCS)
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
            if r ==0
                G(m,n,i) = 1e-4+1j*1e-4;
            end
            for u=1:k3
                p_s=Pos_s(u,:);
                r1=norm(p_tag-p_s);
                r2=norm(p_s-p_recv);
                G(m,n,i)=G(m,n,i)+sqrt(RCS/4/pi/r1^2)*exp(-1j*2*pi/lamda*(r1+r2));
            end
        end
    end
end
end
