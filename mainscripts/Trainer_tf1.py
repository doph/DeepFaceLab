import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
import math
from pathlib import Path
from core import pathex
from core import imagelib
import cv2

import models

from core.interact import interact as io
from core.leras import nn
from core.cv2ex import *
from core.interact import interact as io


def trainerThread (s2c, c2s, e,
                    model_class_name = None,
                    saved_models_path = None,
                    training_data_src_path = None,
                    training_data_dst_path = None,
                    pretraining_data_path = None,
                    pretrained_model_path = None,
                    no_preview=False,
                    force_model_name=None,
                    force_gpu_idxs=None,
                    cpu_only=None,
                    silent_start=False,
                    execute_programs = None,
                    debug=False,
                    target_iter=None,
                    precision=None,
                    bs_per_gpu=None,
                    use_amp=None,
                    opt=None,
                    lr=None,
                    decay_step=None,
                    config_file=None,
                    **kwargs):
    while True:
        try:
            start_time = time.time()

            save_interval_min = 15

            if not training_data_src_path.exists():
                training_data_src_path.mkdir(exist_ok=True, parents=True)

            if not training_data_dst_path.exists():
                training_data_dst_path.mkdir(exist_ok=True, parents=True)

            if not saved_models_path.exists():
                saved_models_path.mkdir(exist_ok=True, parents=True)


            model = models.import_model_tf1(model_class_name)(
                        is_training=True,
                        saved_models_path=saved_models_path,
                        training_data_src_path=training_data_src_path,
                        training_data_dst_path=training_data_dst_path,
                        pretraining_data_path=pretraining_data_path,
                        pretrained_model_path=pretrained_model_path,
                        no_preview=no_preview,
                        force_model_name=force_model_name,
                        force_gpu_idxs=force_gpu_idxs,
                        cpu_only=cpu_only,
                        silent_start=silent_start,
                        debug=debug,
                        target_iter=target_iter,
                        precision=precision,
                        bs_per_gpu=bs_per_gpu,
                        use_amp=use_amp,
                        opt=opt,
                        lr=lr,
                        decay_step=decay_step,
                        config_file=config_file                        
                        )

            is_reached_goal = model.is_reached_iter_goal()

            shared_state = { 'after_save' : False }
            loss_string = ""
            save_iter =  model.get_iter()
            def model_save():
                if not debug and not is_reached_goal:
                    io.log_info ("Saving....", end='\r')
                    model.save()
                    shared_state['after_save'] = True
                    
            def model_backup():
                if not debug and not is_reached_goal:
                    model.create_backup()             

            def send_preview():
                if not debug:
                    previews = model.get_previews()
                    c2s.put ( {'op':'show', 'previews': previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
                else:
                    previews = [( 'debug, press update for new', model.debug_one_iter())]
                    c2s.put ( {'op':'show', 'previews': previews} )
                e.set() #Set the GUI Thread as Ready

            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('Model already trained to target iteration. You can use preview.')
                else:
                    io.log_info('Starting. Target iteration: %d. Press "Enter" to stop training and save model.' % ( model.get_target_iter()  ) )
            else:
                io.log_info('Starting. Press "Enter" to stop training and save model.')

            last_save_time = time.time()

            execute_programs = [ [x[0], x[1], time.time() ] for x in execute_programs ]

            tf = nn.tf
  
            list_globals_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            print('initializing variables ... ')
            list_init = []
            for x in [n for n in list_globals_vars]:
                if 'Adam' in x.name or 'RMSProp' in x.name or 'beta' in x.name or 'loss_scale' in x.name or 'good_steps' in x.name:
                    if not nn.tf_sess.run(nn.tf.is_variable_initialized(x)):
                        list_init.append(x)
            nn.tf_sess.run(tf.variables_initializer(list_init))
            nn.tf_sess.run(model.global_step.initializer)
            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = False)
            print('done ')

            # ( (warped_src, target_src, target_srcm_all), \
            #   (warped_dst, target_dst, target_dstm_all) ) = model.generate_next_samples()

            for i in itertools.count(0,1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if prog_time > 0 and (cur_time - start_time) >= prog_time:
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time)  >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("Unable to execute program: %s" % (prog) )

                    if not is_reached_goal:

                        if model.get_iter() == 0:
                            io.log_info("")
                            io.log_info("Trying to do the first iteration. If an error occurs, reduce the model parameters.")
                            io.log_info("")

                        t_start = time.time()
                        ( (warped_src, target_src, target_srcm_all), \
                          (warped_dst, target_dst, target_dstm_all) ) = model.generate_next_samples()

                        list_loss = []

                        # Train auto-encoder
                        
                        # Train different parts of the network in sequence
                        # More accurate gradient, Slower
                        if True:
                            # _, src_loss, dst_loss, learning_rate = nn.tf_sess.run([model.G_train_op, model.src_loss, model.dst_loss, model.learning_rate], feed_dict={
                            #     model.warped_src :warped_src,
                            #     model.target_src :target_src,
                            #     model.target_srcm_all:target_srcm_all,
                            #     model.warped_dst :warped_dst,
                            #     model.target_dst :target_dst,
                            #     model.target_dstm_all:target_dstm_all})
                            _, src_loss, dst_loss, learning_rate, \
                            gpu_src_code, gpu_src_inter_AB_code, gpu_dst_inter_B_code, \
                            gpu_pred_src_src, gpu_pred_src_srcm = nn.tf_sess.run(
                                [model.G_train_op, model.src_loss, model.dst_loss, model.learning_rate,
                                 model.gpu_src_code, model.gpu_src_inter_AB_code, model.gpu_dst_inter_B_code,
                                 model.gpu_pred_src_src, model.gpu_pred_src_srcm], feed_dict={
                                model.warped_src :warped_src,
                                model.target_src :target_src,
                                model.target_srcm_all:target_srcm_all,
                                model.warped_dst :warped_dst,
                                model.target_dst :target_dst,
                                model.target_dstm_all:target_dstm_all})

                            # print(gpu_src_code.shape)
                            # print(gpu_src_inter_AB_code.shape)
                            # print(gpu_dst_inter_B_code.shape)
                            # print(gpu_pred_src_src.shape)
                            # print(gpu_pred_src_srcm.shape)
                            # print('-----------------------------------------------------')

                            # Train face style
                            if model.options['true_face_power'] != 0:
                                _, _D_code_loss = nn.tf_sess.run([model.D_code_train_op, model.D_code_loss], feed_dict={
                                    model.warped_src :warped_src,
                                    model.target_src :target_src,
                                    model.target_srcm_all:target_srcm_all,
                                    model.warped_dst :warped_dst,
                                    model.target_dst :target_dst,
                                    model.target_dstm_all:target_dstm_all})
                            # Train GAN
                            if model.options['gan_power'] != 0:
                                _, D_src_dst_loss = nn.tf_sess.run([model.D_src_dst_train_op, model.D_src_dst_loss], feed_dict={
                                    model.warped_src :warped_src,
                                    model.target_src :target_src,
                                    model.target_srcm_all:target_srcm_all,
                                    model.warped_dst :warped_dst,
                                    model.target_dst :target_dst,
                                    model.target_dstm_all:target_dstm_all})

                        # Train different parts of the network in parallel
                        # Less accurate gradient, Faster
                        if False:
                            if model.options['gan_power'] == 0 and model.options['true_face_power'] == 0:
                                _G_train_op, src_loss, dst_loss = nn.tf_sess.run(
                                    [model.G_train_op, model.src_loss, model.dst_loss], feed_dict={
                                    model.warped_src :warped_src,
                                    model.target_src :target_src,
                                    model.target_srcm_all:target_srcm_all,
                                    model.warped_dst :warped_dst,
                                    model.target_dst :target_dst,
                                    model.target_dstm_all:target_dstm_all})
                            elif model.options['gan_power'] != 0:
                                _G_train_op, _D_src_dst_train_op, src_loss, dst_loss, D_src_dst_loss = nn.tf_sess.run(
                                    [model.G_train_op, model.D_src_dst_train_op, model.src_loss, model.dst_loss, model.D_src_dst_loss], feed_dict={
                                    model.warped_src :warped_src,
                                    model.target_src :target_src,
                                    model.target_srcm_all:target_srcm_all,
                                    model.warped_dst :warped_dst,
                                    model.target_dst :target_dst,
                                    model.target_dstm_all:target_dstm_all})
                            elif model.options['true_face_power'] != 0:
                                _G_train_op, _D_code_train_op, src_loss, dst_loss, D_code_loss = nn.tf_sess.run(
                                    [model.G_train_op, model.D_code_train_op, model.src_loss, model.dst_loss, model.D_code_loss], feed_dict={
                                    model.warped_src :warped_src,
                                    model.target_src :target_src,
                                    model.target_srcm_all:target_srcm_all,
                                    model.warped_dst :warped_dst,
                                    model.target_dst :target_dst,
                                    model.target_dstm_all:target_dstm_all})                                

                        src_loss = 2.0 if math.isnan(src_loss) else src_loss
                        dst_loss = 2.0 if math.isnan(dst_loss) else dst_loss
                        list_loss = [float(src_loss), float(dst_loss)]


                        model.loss_history.append ( list_loss )
                        model.iter += 1
                        iter_time = time.time() - t_start
                        iter = model.get_iter()


                        if (not io.is_colab() and iter % 200 == 0) or \
                           (io.is_colab() and iter % 200 == 0):
                            plist = []

                            if io.is_colab():
                                previews = model.get_previews()
                                for i in range(len(previews)):
                                    name, bgr = previews[i]
                                    plist += [ (bgr, model.get_strpath_storage_for_file('preview_%s.jpg' % (name) ) ) ]

                            if model.write_preview_history:
                                previews = model.get_static_previews()
                                for i in range(len(previews)):
                                    name, bgr = previews[i]
                                    path = model.preview_history_path / name
                                    path.mkdir(parents=True, exist_ok=True)
                                    plist += [ ( bgr, str ( path / ( f'{iter:07d}.jpg') ) ) ]
                                    if not io.is_colab():
                                        plist += [ ( bgr, str ( path / ( '_last.jpg' ) )) ]
                                                               
                            for preview, filepath in plist:
                                preview_lh = model.get_loss_history_preview(model.loss_history, iter, preview.shape[1], preview.shape[2])
                                img = (np.concatenate ( [preview_lh, preview], axis=0 ) * 255).astype(np.uint8)
                                cv2_imwrite (filepath, img )                            
                        # iter, iter_time = model.train_one_iter()
                        
                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")
                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s]".format ( time_str, iter, '{:0.4f}'.format(iter_time) )
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms]".format ( time_str, iter, int(iter_time*1000) )

                        if shared_state['after_save']:
                            shared_state['after_save'] = False
                            
                            mean_loss = np.mean ( loss_history[save_iter:iter], axis=0)

                            for loss_value in mean_loss:
                                loss_string += "[%.5f]" % (loss_value)

                            io.log_info (loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.5f]" % (loss_value)

                            if io.is_colab():
                                io.log_info ('\r' + loss_string, end='')
                            else:
                                io.log_info (loss_string, end='\r')

                        if model.get_iter() == 1:
                            model_save()

                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info ('\nReached target iteration.')
                            model_save()
                            is_reached_goal = True
                            break
                            io.log_info ('You can use preview now.')

                if not is_reached_goal and (time.time() - last_save_time) >= save_interval_min*60:
                    last_save_time += save_interval_min*60
                    model_save()
                    send_preview()

                if i==0:
                    if is_reached_goal:
                        model.pass_one_iter()
                    send_preview()

                if debug:
                    time.sleep(0.005)

                while not s2c.empty():
                    input = s2c.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'backup':
                        model_backup()
                    elif op == 'preview':
                        if is_reached_goal:
                            model.pass_one_iter()
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break

                if i == -1:
                    break

            model.finalize()



        except Exception as e:
            print ('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    c2s.put ( {'op':'close'} )


def main(**kwargs):
    io.log_info ("Running trainer (TF1).\r\n")

    no_preview = kwargs.get('no_preview', False)

    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e), kwargs=kwargs )
    thread.start()

    e.wait() #Wait for inital load to occur.

    if no_preview:
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'close':
                    break
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )
    else:
        wnd_name = "Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0
        iter = 0
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iter = input['iter'] if 'iter' in input.keys() else 0
                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            (h, w, c) = preview_rgb.shape
                            max_h = max (max_h, h)
                            max_w = max (max_w, w)

                        max_size = 800
                        if max_h > max_size:
                            max_w = int( max_w / (max_h / max_size) )
                            max_h = max_size

                        #make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append ( (preview_name, cv2.resize(preview_rgb, (max_w, max_h))) )
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False

                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                (h,w,c) = selected_preview_rgb.shape

                # HEAD
                head_lines = [
                    '[s]:save [b]:backup [enter]:exit',
                    '[p]:update [space]:next preview [l]:change history range',
                    'Preview: "%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
                    ]
                head_line_height = 15
                head_height = len(head_lines) * head_line_height
                head = np.ones ( (head_height,w,c) ) * 0.1

                for i in range(0, len(head_lines)):
                    t = i*head_line_height
                    b = (i+1)*head_line_height
                    head[t:b, 0:w] += imagelib.get_text_image (  (head_line_height,w,c) , head_lines[i], color=[0.8]*c )

                final = head

                if loss_history is not None:
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]

                    lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)
                    final = np.concatenate ( [final, lh_img], axis=0 )

                final = np.concatenate ( [final, selected_preview_rgb], axis=0 )
                final = np.clip(final, 0, 1)

                io.show_image( wnd_name, (final*255).astype(np.uint8) )
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == ord('\n') or key == ord('\r'):
                s2c.put ( {'op': 'close'} )
            elif key == ord('s'):
                s2c.put ( {'op': 'save'} )
            elif key == ord('b'):
                s2c.put ( {'op': 'backup'} )
            elif key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put ( {'op': 'preview'} )
            elif key == ord('l'):
                if show_last_history_iters_count == 0:
                    show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                    show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0
                update_preview = True
            elif key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True

            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )

        io.destroy_all_windows()
