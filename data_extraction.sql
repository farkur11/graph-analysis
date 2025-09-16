
drop table if exists payme_sandbox.transfers_direction_temp;
create table payme_sandbox.transfers_direction_temp as
select payer_id, card_number sender_card, sendercard_phone payer_phone,
       regexp_replace(trim(sendercard_owner), '\\s+', ' ', 'g') sender_name,
       date_trunc('minute', create_time)::timestamp ds, sensitive_data_balance_before_payment/100 balance_before,
       amount/100 amount, account_number receiver_card,
       regexp_replace(trim(account_cardowner), '\\s+', ' ', 'g') receiver_name
from ods__mdbmn__paycom.receipts r
where r."state" = '4'
and r."type" = '5'
and r.payment_service = '56e7ce796b6ef347d846e3eb'
and r.external = false
and r.meta_owner is null
and create_time < current_date
and create_time > current_date - interval '150 days'