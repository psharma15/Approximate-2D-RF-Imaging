%% This is the newest, correct forward model for maxent by 07th Feb 2017

function maxent_fwdmod()
    c = physconst('LightSpeed');
    f = 'f3';
    Ntag = 16; 
    Nrecv = 4;

    switch(f)     
        case 'f3'
            Freq =(1.7:0.05:2.2)*1e9;               
        case 'f5'
            Freq = [1.79 2.03 2.2]*1e9; % 0.9 1.5 2.2
        case 'f6'
            Freq = 2e9;
        otherwise
            display('Error in frequency');
    end
    Nfreq = length(Freq);
    Pos_tag = zeros(Ntag,2);
    switch(Ntag)
        case 32
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
    Pos_recv = zeros(Nrecv,2);
    switch(Nrecv)
        case 1
            Pos_recv = [-0.5,-0.5];
        case 2
            Pos_recv = [-0.5,-0.5;0.5,-0.5];   
        case 4
            Pos_recv = [-0.5,-0.5;0.5,-0.5;0.5,0.5;-0.5,0.5];
        case 6
            Pos_recv(:,1) = linspace(-0.5,0.5,6);
            Pos_recv(:,2) = ones(Nrecv,1).*-0.5;
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

        otherwise
            disp('Error in Pos_recv');
    end
    figure;
    plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
    hold on;
    plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);

    x_v = linspace(-0.4,0.4,40);
    y_v = x_v;
    lx = length(x_v);
    [X, Y] = meshgrid(x_v,y_v);
    P = combvec(x_v,y_v)';

    R1 = repmat(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Ntag 1]) ...
    - repmat(permute(Pos_tag, [3 1 2]), [lx*lx 1 1]) ).^2, 3)),1,Nrecv);

    dummy1 = ones(1,Ntag);
    R2 = kron(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Nrecv 1]) ...
    - repmat(permute(Pos_recv, [3 1 2]), [lx*lx 1 1]) ).^2, 3)), dummy1);

    R = R1 + R2; 
    R = repmat(R,1,Nfreq); % All tag, recv, freq
    dummy3 = ones(1,Ntag*Nrecv);
    Freq0 = repmat(kron(Freq,dummy3),[lx*lx 1]);

    e = exp(-1j*(2*pi/c)*(Freq0.*R));
    e = transpose(e); % == A matrix
    [m,n] = size(e);

    %% Generate G from forward model and verify with Fourier inverse 
    img = zeros(lx,lx);
    img(18:22,18:22) = 1;
    figure;
    mesh(X,Y,img'); colormap('gray')
    x = reshape(img,[],1);
    std_dev = 0.05;
    % e*x is NOT exact error-free right forward model: discretized, real
    % object reflectivity
    G = e*x + std_dev*randn(m,1); 
    B = diag(e'*(G*G')*e);
    B = real(B);
    B = (reshape(B, [lx lx]));
    figure
    mesh(X,Y,B')
    axis 'tight'
    axis 'square'
    
    %% Clear temp variables 
    clearvars -except e m n g G R Ntag Nrecv Freq lx  ncomb c X Y
    
    %% get forward model for Maxent after taking autocorrelation
    E1 = e';
    E2 = transpose(e);
    E3 = zeros(n,m*m);
    j = 1;
    for i = 1:m:m*m
        E3(:,i:i+m-1) = repmat(E1(:,j),[1,m]).*E2;
        j = j+1;
    end
    E3 = E3/sum(E3(:));
    v = transpose(G*G'); % E3*v = image
    v = reshape(v,[],1);
    B1 = E3*v;
    gamma = sum(B1)
    B1 = real(B1);
    B1 = (reshape(B1, [lx lx]));
    figure
    mesh(X,Y,B1')
    axis 'tight'
    axis 'square'
    g = E3';
%     clearvars -except v g
    v = [real(v); imag(v)];
    g = [real(g); imag(g)];
    assignin('base','g',g);
    assignin('base','v',v); % g*image = V -? Right forward model
   
end

