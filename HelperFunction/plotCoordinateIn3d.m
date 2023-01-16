load('save_data_AVG0.mat', 'pred')
Color = parula(size(pred,1));

% Links = [1,2;
%     2,3;
%     3,1;
%     1,4;
%     4,5;
%     5,6;
%     4,7;
%     4,8
%     6,9;
%     6,10];

Links =  [1,2;
          1,3;
          2,3;
          3,4;
          4,5;
          5,6;
          4,11;
          4,12;
          11,9;
          9,10;
          12,13;
          13,14;
          6,7;
          7,16;
          16,17;
          6,8;
          8,15;
          15,18];

% figure;
% xlim([-400,400])
% ylim([-400,400])
% zlim([0,400])
view(55,45);
axis([-400,400,-400,400,0,400])
for i = 1:100:size(pred,1)
    i
    subplot(1,1,1); hold on;
    % plot3(squeeze(pred(i,1,:)),squeeze(pred(i,2,:)),squeeze(pred(i,3,:)),'o','color',Color(i,:))
    for k = 1:size(Links)
        X = [pred(i,1,Links(k,1)) pred(i,1,Links(k,2))] ;
        Y = [pred(i,2,Links(k,1)) pred(i,2,Links(k,2))] ;
        Z = [pred(i,3,Links(k,1)) pred(i,3,Links(k,2))] ;
        plot3(X,Y,Z,'-','color',Color(i,:));
    end
    pause(1)
    hold off;
end

%%
% v = VideoWriter('3dtracking.avi');
% v.FrameRate = 25;
% open(v);

load('save_data_AVG0.mat', 'pred')
Color = parula(size(pred,1));

% Links = [1,2;
%     2,3;
%     3,1;
%     1,4;
%     4,5;
%     5,6;
%     4,7;
%     4,8];

Links =  [1,2;
          1,3;
          2,3;
          3,4;
          4,5;
          5,6;
          4,11;
          4,12;
          11,9;
          9,10;
          12,13;
          13,14;
          6,7;
          7,16;
          16,17;
          6,8;
          8,15;
          15,18];

grid on;
h = animatedline('MaximumNumPoints', 16);

Color2 = parula(8);
% Force a 3D view
view(55,45);
axis([-400,400,-400,400,0,400])
for i = 1:size(pred,1)
    for k = 1:size(Links)
        X = [pred(i,1,Links(k,1)) pred(i,1,Links(k,2))] ;
        Y = [pred(i,2,Links(k,1)) pred(i,2,Links(k,2))] ;
        Z = [pred(i,3,Links(k,1)) pred(i,3,Links(k,2))] ;
        addpoints(h,X,Y,Z);
    end
    h.Color = Color(i,:);
    drawnow
%     frame = getframe(gcf);
%     writeVideo(v,frame);
%     pause(0.1)
end

% close(v)
