output_directory_path = 'C:\Users\Sara\Desktop\FullDatasetLoopedPN100%Again';
RAVDESS_path= 'C:\Users\Sara\Desktop\FullDataset_Looped';

files = dir(fullfile(RAVDESS_path, '*.*'));
% Exclude directories from the list
files = files(~[files.isdir]);

for i = 1:numel(files)
    current_file = fullfile(RAVDESS_path, files(i).name);
    [data, fs] = audioread(current_file); 
    
    processed_data = data + 0.1 * pinknoise(size(data, 1), 1);
    
    % Scale the processed data to fit within the valid range
    max_value = max(abs(processed_data));
    if max_value > 1
        processed_data = processed_data / max_value;
    end
    
    [~, file_name, file_extension] = fileparts(current_file);
    processed_file_name = [file_name, '_processed', file_extension];
    processed_file_path = fullfile(output_directory_path, processed_file_name);
    
    audiowrite(processed_file_path, processed_data, fs);
end
